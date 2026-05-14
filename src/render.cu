//===================================================================
//render.cu - CUDA光栅化+内存管理
//===================================================================
#include "render.h"
#include <cuda_runtime.h>				//CUDA运行时API
#include <device_launch_parameters.h>	//设备启动参数，threadIdx，blockIdx等
#include <cfloat>

static unsigned int* g_dFramebuf = nullptr;	//设备端帧缓冲区指针（BGRA像素）
static float* g_dDepthbuf = nullptr;		//设备端深度缓冲区指针
static GPUTriangle* g_dTris = nullptr;		//设备端三角形数据指针
static int g_numTris = 0;					//当前三角形数量
static int g_fbWidth = 0;					//帧缓冲宽度
static int g_fbHeight = 0;					//帧缓冲高度
static TriCached* g_dTriCache = nullptr;	//一改，三角形NDC坐标+包围盒
static TileData* g_dTiles = nullptr;		//二改，分块里的三角形索引和数量
static int g_tilesX = 0, g_tilesY = 0;		//二改，X和Y方向的分块数
static BVHNode* g_dBVH = nullptr;			// 三改：设备端 BVH 节点数组指针
static int      g_numBVHNodes = 0;			// 三改：设备端 BVH 节点数量

//边函数，屏幕向下看，判断点在线的左or右(正顺右，负逆左)，返回有符号面积（正向屏上，负向屏下）
__device__ __forceinline__
float edgeFunc(float ax, float ay, float bx, float by, float px, float py) {
	return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

//一改：提前算好每个三角形的ndc坐标，像素包围盒和基础光照，逐像素时拿现成的判断+计算高光后的光照
__global__ void computeTriCacheKernel(
	TriCached* triCache,
	const GPUTriangle* tris,
	int numTris,
	RenderParams p)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;	//三角形编号
	if (t >= numTris) return;						//多余线程跳过

	const GPUTriangle& tri = tris[t];				//要处理的三角形
	TriCached& tc = triCache[t];					//该三角形的ndc，包围盒，基础光照

	float minX = 1e30f, maxX = -1e30f;
	float minY = 1e30f, maxY = -1e30f;

	for (int k = 0; k < 3; ++k) {
		float vx = tri.v[k].x, vy = tri.v[k].y, vz = tri.v[k].z;	//三顶点模型空间坐标

		float cx = p.mvp.m[0][0] * vx + p.mvp.m[1][0] * vy + p.mvp.m[2][0] * vz + p.mvp.m[3][0];
		float cy = p.mvp.m[0][1] * vx + p.mvp.m[1][1] * vy + p.mvp.m[2][1] * vz + p.mvp.m[3][1];
		float cz = p.mvp.m[0][2] * vx + p.mvp.m[1][2] * vy + p.mvp.m[2][2] * vz + p.mvp.m[3][2];
		float cw = p.mvp.m[0][3] * vx + p.mvp.m[1][3] * vy + p.mvp.m[2][3] * vz + p.mvp.m[3][3];

		if (cw <= 0.f) {
			tc.nx[k] = -2.f; tc.ny[k] = -2.f; tc.nz[k] = 2.f;		//设为点在屏幕外
			minX = -2.f; maxX = -2.f; minY = -2.f; maxY = -2.f;		//包围盒在屏幕外，跳过
			continue;
		}
		//透视除法
		tc.nx[k] = cx / cw;
		tc.ny[k] = cy / cw;
		tc.nz[k] = cz / cw;
		//ndc空间的包围盒
		minX = fminf(minX, tc.nx[k]);
		maxX = fmaxf(maxX, tc.nx[k]);
		minY = fminf(minY, tc.ny[k]);
		maxY = fmaxf(maxY, tc.ny[k]);
	}
	//屏幕像素包围盒
	tc.minX = fmaxf(0, fminf(p.width - 1, (int)((minX + 1.0f) * 0.5f * p.width) - 1));
	tc.maxX = fmaxf(0, fminf(p.width - 1, (int)((maxX + 1.0f) * 0.5f * p.width) + 1));
	tc.minY = fmaxf(0, fminf(p.height - 1, (int)((1.0f - maxY) * 0.5f * p.height)));
	tc.maxY = fmaxf(0, fminf(p.height - 1, (int)((1.0f - minY) * 0.5f * p.height)));

	// 计算背面标记和面积（用于后面判断）
	float area = edgeFunc(tc.nx[0], tc.ny[0], tc.nx[1], tc.ny[1], tc.nx[2], tc.ny[2]);
	tc.flipped = (area < 0.f);
	tc.area = fabsf(area);

	Vec3 N = tri.normal;				//三角形面法线
	if (tc.flipped) { N = -N; }			//背面光照：使用相反法线
	tc.normal = N;						//缓存法线

	//基础光计算
	//提前把0-255转换成0.0-1.0
	float baseR = tri.color[0] / 255.f;  //基础红色分量
	float baseG = tri.color[1] / 255.f;  //基础绿色分量
	float baseB = tri.color[2] / 255.f;  //基础蓝色分量

	//漫反射
	Vec3 L = p.lightDir;				 //光照方向,指向光源
	float keyDiff = fmaxf(N.dot(L), 0.f);//漫反射系数（值越大越亮，范围0-1，负数被0.f钳制到0）

	//补光
	Vec3 fillDir = { -0.4f, -0.3f, 0.6f };					// 补光方向
	float fillDiff = fmaxf(N.dot(fillDir.norm()), 0.f);		// 补光漫反射系数

	//半球环境光,朝上亮，朝下暗
	float hemi = 0.7f + 0.3f * N.y;

	// 预计算基础色（高光除外，因为高光依赖视角/像素位置）
	tc.baseR = fminf(1.f,								//光照为1或者
		p.ambient.x * (0.8f + 0.2f * hemi) * baseR		//环境光
		+ keyDiff * p.lightColor.x * baseR * 0.85f		//漫反射主光
		+ fillDiff * 0.35f * baseR						//补光
	);
	tc.baseG = fminf(1.f,
		p.ambient.y * (0.8f + 0.2f * hemi) * baseG
		+ keyDiff * p.lightColor.y * baseG * 0.85f
		+ fillDiff * 0.3f * baseG
	);
	tc.baseB = fminf(1.f,
		p.ambient.z * (0.8f + 0.2f * hemi) * baseB
		+ keyDiff * p.lightColor.z * baseB * 0.85f
		+ fillDiff * 0.4f * baseB
	);
}
// 三改：将 BVH 节点的模型空间 AABB 经 MVP 矩阵保守投影到屏幕像素空间
// 返回 true  表示节点至少部分位于视锥内（可能在屏幕上可见）
// 返回 false 表示 8 个角点全部在相机后方，可安全剔除
__device__ bool nodeScreenBounds(const BVHNode& node, const Mat4& mvp,
	int width, int height,
	int& outMinX, int& outMaxX, int& outMinY, int& outMaxY)
{
	float minX = 1e30f, maxX = -1e30f;
	float minY = 1e30f, maxY = -1e30f;
	bool anyFront = false;

	// AABB 的 8 个角点坐标（模型空间）
	float cx[8] = { node.minX, node.maxX, node.minX, node.maxX,
					node.minX, node.maxX, node.minX, node.maxX };
	float cy[8] = { node.minY, node.minY, node.maxY, node.maxY,
					node.minY, node.minY, node.maxY, node.maxY };
	float cz[8] = { node.minZ, node.minZ, node.minZ, node.minZ,
					node.maxZ, node.maxZ, node.maxZ, node.maxZ };

	for (int i = 0; i < 8; ++i) {
		float x = cx[i], y = cy[i], z = cz[i];
		// MVP 变换到裁剪空间
		float wx = mvp.m[0][0] * x + mvp.m[1][0] * y + mvp.m[2][0] * z + mvp.m[3][0];
		float wy = mvp.m[0][1] * x + mvp.m[1][1] * y + mvp.m[2][1] * z + mvp.m[3][1];
		float wz = mvp.m[0][2] * x + mvp.m[1][2] * y + mvp.m[2][2] * z + mvp.m[3][2];
		float ww = mvp.m[0][3] * x + mvp.m[1][3] * y + mvp.m[2][3] * z + mvp.m[3][3];

		if (ww <= 0.0f) continue;   // 角点在相机后方，保守跳过（不扩展边界）
		anyFront = true;

		// 透视除法 -> NDC -> 屏幕像素（Y 轴翻转，使左上角为原点）
		float ndcX = wx / ww;
		float ndcY = wy / ww;
		float sx = (ndcX + 1.0f) * 0.5f * width;
		float sy = (1.0f - ndcY) * 0.5f * height;

		minX = fminf(minX, sx); maxX = fmaxf(maxX, sx);
		minY = fminf(minY, sy); maxY = fmaxf(maxY, sy);
	}

	if (!anyFront) return false; // 全部在相机后方，视为不可见

	// 向外扩展 1 像素，确保保守性
	outMinX = fmaxf(0, (int)floorf(minX) - 1);
	outMaxX = fminf(width - 1, (int)ceilf(maxX) + 1);
	outMinY = fmaxf(0, (int)floorf(minY) - 1);
	outMaxY = fminf(height - 1, (int)ceilf(maxY) + 1);
	return true;
}
//二改：分块构建内核（三改：BVH剔除远三角形分支）
__global__ void buildTileKernel(
	TileData* tiles,
	const TriCached* triCache,
	int numTris,
	int tilesX, int tilesY,
	int width, int height,
	const Mat4 mvp,				// 三改：MVP 矩阵，用于将 BVH 节点投影到屏幕
	const BVHNode* bvhNodes,    // 三改参数
	int numBVHNodes)            // 三改参数
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx >= tilesX || ty >= tilesY) return;

	int tileIdx = ty * tilesX + tx;
	TileData& tile = tiles[tileIdx];
	tile.triCount = 0;

	// 当前块的像素范围
	int tileMinX = tx * TILE_SIZE;
	int tileMaxX = fminf((tx + 1) * TILE_SIZE - 1, width - 1);
	int tileMinY = ty * TILE_SIZE;
	int tileMaxY = fminf((ty + 1) * TILE_SIZE - 1, height - 1);

	// 若 BVH 未上传（兼容旧逻辑），回退到暴力遍历，找出覆盖该块的三角形
	if (!bvhNodes || numBVHNodes == 0) {
	for (int t = 0; t < numTris && tile.triCount < MAX_TRIS_PER_TILE; ++t) {
		const TriCached& tc = triCache[t];

		// 检查三角形包围盒是否和块相交
		if (tc.maxX < tileMinX || tc.minX > tileMaxX ||
			tc.maxY < tileMinY || tc.minY > tileMaxY)
			continue;

		tile.triIdx[tile.triCount++] = t;
	}
	return;
}// BVH 栈遍历：只访问可能与该 Tile 屏幕区域相交的节点
	int stack[64];   // 显式栈，深度足以容纳 2^64 节点（实际 log2(N) 通常 < 25）
	int sp = 0;
	stack[sp++] = 0; // 根节点索引固定为 0

	while (sp > 0 && tile.triCount < MAX_TRIS_PER_TILE) {
		int nodeIdx = stack[--sp];
		const BVHNode& node = bvhNodes[nodeIdx];

		// 1. 计算该节点在屏幕上的保守包围盒
		int nMinX, nMaxX, nMinY, nMaxY;
		if (!nodeScreenBounds(node, mvp, width, height, nMinX, nMaxX, nMinY, nMaxY))
			continue; // 完全在视锥外或相机后方，整棵子树剔除

		// 2. 节点包围盒与 Tile 无交集？剪枝
		if (nMaxX < tileMinX || nMinX > tileMaxX ||
			nMaxY < tileMinY || nMinY > tileMaxY)
			continue;

		if (node.triCount > 0) {
			// 3. 叶子节点：用精确的三角形屏幕包围盒二次过滤后加入 Tile
			for (int i = 0; i < node.triCount && tile.triCount < MAX_TRIS_PER_TILE; ++i) {
				int t = node.triStart + i;
				if (t < 0 || t >= numTris) continue; // 防御性检查
				const TriCached& tc = triCache[t];
				if (tc.maxX < tileMinX || tc.minX > tileMaxX ||
					tc.maxY < tileMinY || tc.minY > tileMaxY)
					continue;
				tile.triIdx[tile.triCount++] = t;
			}
		}
		else {
			// 4. 内部节点：将子节点压栈（先右后左，使左子树先被处理）
			if (node.right >= 0) stack[sp++] = node.right;
			if (node.left >= 0) stack[sp++] = node.left;
		}
	}
}

//主光栅化，对于每个像素
__global__
void rasterizeKernel(
	unsigned int* framebuf,
	float* depthbuf,
	const TriCached* triCache,  // 一改,三角形ndc，包围盒，基础光数组
	const GPUTriangle* tris,
	const TileData* tiles,		// 二改，块数组，存块里的三角形索引和数量
	int tilesX,					// 二改，X方向分块数量
	RenderParams p
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;//屏幕x坐标像素
	int py = blockIdx.y * blockDim.y + threadIdx.y;//屏幕y坐标像素
	if (px >= p.width || py >= p.height)return;//跳过显示区域外的多余线程的像素处理

	// 找到这个像素属于哪个块
	int tx = px / TILE_SIZE;
	int ty = py / TILE_SIZE;
	int tileIdx = ty * tilesX + tx;
	const TileData& tile = tiles[tileIdx];
	// 背景块：没有三角形覆盖,直接返回
	if (tile.triCount == 0) return;

	int idx = py * p.width + px;//像素索引

	//+0.5选像素中心，然后归一化到NDC空间
	float ndcX = (px + 0.5f) / p.width * 2.f - 1.f;
	float ndcY = 1.f - (py + 0.5f) / p.height * 2.f;

	float bestZ = depthbuf[idx];			//最前边的深度值，开始是FLT_MAX
	unsigned int bestColor = framebuf[idx];	//最前边的物体的颜色，开始是clear画的灰色

	//定一个像素，挨个看三角形
	//只遍历这个块里的三角形
	for (int i = 0; i < tile.triCount; ++i) {
		int t = tile.triIdx[i];
		const TriCached& tc = triCache[t];
		//包围盒快速剔除
		if (px < tc.minX || px > tc.maxX || py < tc.minY || py > tc.maxY)
			continue;

		// 退化三角形剔除（面积为零，跳过）
		if (tc.area == 0.f) continue;

		float nx0 = tc.nx[0], ny0 = tc.ny[0], nz0 = tc.nz[0];
		float nx1 = tc.nx[1], ny1 = tc.ny[1], nz1 = tc.nz[1];
		float nx2 = tc.nx[2], ny2 = tc.ny[2], nz2 = tc.nz[2];

		if (tc.flipped) {
			// 交换顶点0和顶点2的标签，翻转面朝向，保证点都顺时针，便于用w>=0判断在内部
			float tmp;
			tmp = nx0; nx0 = nx2; nx2 = tmp;  // 交换X坐标
			tmp = ny0; ny0 = ny2; ny2 = tmp;  // 交换Y坐标
			tmp = nz0; nz0 = nz2; nz2 = tmp;  // 交换Z坐标
		}

		//细：重心坐标计算比较（权重和为1，而且面积占的越大离相应点越近，w也能反映距离）
		float w0 = edgeFunc(nx1, ny1, nx2, ny2, ndcX, ndcY) / tc.area;	//顶点 V0 的权重 = P 到对边 V1-V2 的距离比例
		float w1 = edgeFunc(nx2, ny2, nx0, ny0, ndcX, ndcY) / tc.area;	//顶点 V1 的权重 = P 到对边 V2-V0 的距离比例
		float w2 = 1.f - w0 - w1;									//剩下的权重
		if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;				//有负，则在三角形外部，跳过该三角形

		//线框模式
		if (p.wireframe) {
			float thresh = 0.025f;	//线框宽度
			if (w0 > thresh && w1 > thresh && w2 > thresh) continue;//离框太远不渲染
		}
		//深度测试
		float z = w0 * nz0 + w1 * nz1 + w2 * nz2;	//三点深度换成三角形深度
		if (z >= bestZ) continue;   //被更近的遮挡，不渲染
		bestZ = z;                  // 更新最佳深度值

		//Blinn_Phong光照
		const GPUTriangle& tri = tris[t];
		Vec3 N = tc.normal;
		//高光
		Vec3 L = p.lightDir;
		Vec3 V = { 0.f,0.f,1.f };	//观察方向，看向屏幕外
		Vec3 H = (L + V).norm();	//半程向量
		float spec = powf(fmaxf(N.dot(H), 0.f), p.specPower);  // 高光强度（指数控制高光集中度）
		//合成光照，3通道分别合成
		float r = fminf(1.f, tc.baseR + spec * 0.4f);
		float g = fminf(1.f, tc.baseG + spec * 0.4f);
		float b = fminf(1.f, tc.baseB + spec * 0.4f);
		//打包成BGRA（Windows DIB标准）,浮点RGB的0-1转为32位二进制,用unsigned int位运算
		bestColor = 0xFF000000u					// 透明度FF决定不透明（最高位）
			| ((unsigned int)(b * 255) << 16)   // 蓝色分量（高位）
			| ((unsigned int)(g * 255) << 8)	// 绿色分量（中位）
			| ((unsigned int)(r * 255));        // 红色分量（低位）
	}
	//结果返回输出缓冲区
	depthbuf[idx] = bestZ;		//深度
	framebuf[idx] = bestColor;	//颜色
}

//清屏内核（背景布）
__global__
void clearKernel(unsigned int* framebuf, float* depthbuf, int width, int height)
{
	// 计算像素坐标
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= width || py >= height) return;  // 边界检查

	int idx = py * width + px;  // 计算线性索引

	// 垂直渐变：根据屏幕Y坐标计算灰度值
	float t = (float)py / height;                     // 归一化Y坐标 [0, 1]
	unsigned char bv = (unsigned char)(12 + t * 30);  // 顶部暗灰(RGB≈12)，底部亮灰(RGB≈42)
	framebuf[idx] = 0xFF000000u | (bv << 16) | (bv << 8) | bv;
	// 初始化深度缓冲区
	depthbuf[idx] = FLT_MAX;
}


//导出DLL文件的API
//初始化CUDA
extern "C" CUDA_API
int cudaInit(int width, int height, char* outName, int nameBufSize) {
	//检测可用GPU数量
	int devCount = 0;
	cudaGetDeviceCount(&devCount);
	if (devCount == 0)return -1;	//无可用GPU，返回失败

	cudaSetDevice(0);		//选0号GPU
	if (outName && nameBufSize > 0) {
		cudaDeviceProp prop;				//GPU属性结构体
		cudaGetDeviceProperties(&prop, 0);	//获取0的属性
		strncpy(outName, prop.name, nameBufSize - 1);
		outName[nameBufSize - 1] = '\0';	//防止名字溢出（基本不会出现，但以防万一）
	}

	g_fbWidth = width;
	g_fbHeight = height;
	size_t sz = (size_t)width * height;	//总像素数
	int tilesX = (width + TILE_SIZE - 1) / TILE_SIZE;
	int tilesY = (height + TILE_SIZE - 1) / TILE_SIZE;
	g_tilesX = tilesX;
	g_tilesY = tilesY;

	// 释放可能已存在的旧缓冲区
	if (g_dFramebuf) cudaFree(g_dFramebuf);
	if (g_dDepthbuf) cudaFree(g_dDepthbuf);
	if (g_dTriCache) cudaFree(g_dTriCache);
	if (g_dTiles) cudaFree(g_dTiles);
	// 分配新的GPU帧缓冲区和深度缓冲区
	cudaError_t err1 = cudaMalloc(&g_dFramebuf, sz * sizeof(unsigned int));  // 帧缓冲（BGRA像素）
	cudaError_t err2 = cudaMalloc(&g_dDepthbuf, sz * sizeof(float));         // 深度缓冲（浮点）
	cudaError_t err3 = cudaMalloc(&g_dTriCache, 500000 * sizeof(TriCached)); // 预处理数据缓冲，最大50万三角形
	cudaError_t err4 = cudaMalloc(&g_dTiles, tilesX * tilesY * sizeof(TileData));//分块内存
	// 检查内存分配是否成功
	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) return -1;
	return 0;  // 成功
}

// 上传 BVH 节点数据到 GPU 显存
extern "C" CUDA_API
void cudaUploadBVH(const BVHNode* nodes, int numNodes) {
	if (g_dBVH) {
		cudaFree(g_dBVH);
		g_dBVH = nullptr;
	}
	g_numBVHNodes = numNodes;

	if (numNodes > 0 && nodes) {
		size_t bytes = (size_t)numNodes * sizeof(BVHNode);
		cudaError_t err = cudaMalloc(&g_dBVH, bytes);
		if (err == cudaSuccess) {
			cudaMemcpy(g_dBVH, nodes, bytes, cudaMemcpyHostToDevice);
		}
		else {
			g_dBVH = nullptr;
			g_numBVHNodes = 0;
		}
	}
}

// 上传三角形数据到GPU
extern "C" CUDA_API
void cudaUploadTriangles(const GPUTriangle* tris, int numTris) {
	if (g_dTris) {
		cudaFree(g_dTris);     // 释放GPU显存
		g_dTris = nullptr;     // 重置指针
	}
	g_numTris = numTris;       // 更新三角形计数

	// 如果有新数据需要上传
	if (numTris > 0 && tris) {
		size_t bytes = (size_t)numTris * sizeof(GPUTriangle);  // 计算字节数
		cudaError_t err = cudaMalloc(&g_dTris, bytes);         // 分配GPU显存
		if (err != cudaSuccess) {
			// 分配失败：重置状态
			g_dTris = nullptr;
			g_numTris = 0;
			return;
		}
		// 将三角形数据从主机内存拷贝到设备显存
		cudaMemcpy(g_dTris, tris, bytes, cudaMemcpyHostToDevice);
	}
}

//渲染一帧
extern "C" CUDA_API
void cudaRender(const RenderParams& params, unsigned int* g_hFramebuf) {
	// 检查GPU缓冲区和输出指针是否有效
	if (!g_dFramebuf || !g_dDepthbuf || !g_hFramebuf) return;
	if (!g_dTris || !g_dTriCache || !g_dTiles) return;
	// 配置CUDA线程网格（每个线程块16×16线程）
	dim3 block(16, 16);
	//算需要线程块数，向上取整
	dim3 grid((params.width + block.x - 1) / block.x,
		(params.height + block.y - 1) / block.y);
	//第一步：清屏
	clearKernel << <grid, block >> > (g_dFramebuf, g_dDepthbuf, params.width, params.height);
	cudaDeviceSynchronize();  // 等待清屏完成
	//第二步：GPU预计算每个三角形的NDC坐标+屏幕包围盒
	int triBlocks = (params.numTris + 255) / 256;
	computeTriCacheKernel << <triBlocks, 256 >> > (g_dTriCache, g_dTris, params.numTris, params);
	cudaDeviceSynchronize();  // 等待预计算完成
	//第三步：构建分块
	int tilesX = (params.width + TILE_SIZE - 1) / TILE_SIZE;
	int tilesY = (params.height + TILE_SIZE - 1) / TILE_SIZE;
	dim3 tileGrid((tilesX + 15) / 16, (tilesY + 15) / 16);
	buildTileKernel << <tileGrid, dim3(16, 16) >> > (
		g_dTiles, 
		g_dTriCache, 
		params.numTris,
		tilesX, tilesY, 
		params.width, params.height, 
		params.mvp,         // 三改：传入 MVP 矩阵供 BVH 投影使用
		g_dBVH,				// 三改:传入设备端 BVH 指针
		g_numBVHNodes);		// 三改:传入节点数量
	cudaDeviceSynchronize();
	//第四步：光栅化三角形
	rasterizeKernel << <grid, block >> > (g_dFramebuf, g_dDepthbuf,
		g_dTriCache,
		g_dTris ? g_dTris : nullptr,  // 如果没有三角形，传nullptr
		g_dTiles,
		tilesX,
		params);
	cudaDeviceSynchronize();
	//第五步：显存结果拷贝回内存
	cudaMemcpy(g_hFramebuf, g_dFramebuf,
		(size_t)params.width * params.height * sizeof(unsigned int),
		cudaMemcpyDeviceToHost);
}

//释放GPU
extern "C" CUDA_API
void cudaShutdown(void)
{
	// 释放帧缓冲区
	if (g_dFramebuf) {
		cudaFree(g_dFramebuf);
		g_dFramebuf = nullptr;
	}
	// 释放深度缓冲区
	if (g_dDepthbuf) {
		cudaFree(g_dDepthbuf);
		g_dDepthbuf = nullptr;
	}
	// 释放三角形数据
	if (g_dTris) {
		cudaFree(g_dTris);
		g_dTris = nullptr;
	}
	// 释放预处理缓存
	if (g_dTriCache) {
		cudaFree(g_dTriCache);
		g_dTriCache = nullptr;
	}
	// 释放分块
	if (g_dTiles) {
		cudaFree(g_dTiles);
		g_dTiles = nullptr;
	}
	// 释放 BVH 节点
	if (g_dBVH) {
		cudaFree(g_dBVH);
		g_dBVH = nullptr;
	}
	g_numBVHNodes = 0;

	g_numTris = 0;  // 重置三角形计数
}
