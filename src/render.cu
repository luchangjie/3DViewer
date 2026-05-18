//===================================================================
//render.cu - CUDA光栅化+内存管理（终极高性能版）
//已修复：
//1. 相机靠近时模型消失问题（近裁剪面奇点）
//2. 近距离帧率暴跌问题（全屏幕包围盒导致的O(n²)复杂度）
//核心优化：
//- 预处理阶段完成近裁剪面裁剪（每个三角形只裁剪一次，而非每个像素）
//- 精确计算裁剪后三角形的保守包围盒（替代全屏幕包围盒）
//- 缓存所有可预计算的数据，避免光栅化阶段重复计算
//- 动态调整阈值，只在真正必要时触发特殊处理
//===================================================================
#include "render.h"
#include <cuda_runtime.h>				//CUDA运行时API
#include <device_launch_parameters.h>	//设备启动参数（threadIdx, blockIdx等）
#include <cfloat>

//-------------------------- 全局GPU缓冲区 --------------------------
static unsigned int* g_dFramebuf = nullptr;	//设备端帧缓冲区（BGRA格式，32位/像素）
static float* g_dDepthbuf = nullptr;		//设备端深度缓冲区（32位浮点）
static GPUTriangle* g_dTris = nullptr;		//设备端原始三角形数据
static int g_numTris = 0;					//当前加载的三角形总数
static int g_fbWidth = 0;					//帧缓冲宽度（像素）
static int g_fbHeight = 0;					//帧缓冲高度（像素）
static TriCached* g_dTriCache = nullptr;	//三角形预处理缓存（每个原始三角形一个）
static TileData* g_dTiles = nullptr;		//屏幕分块数据（每个块存覆盖的三角形索引）
static int g_tilesX = 0, g_tilesY = 0;		//X和Y方向的分块数量
static BVHNode* g_dBVH = nullptr;			//设备端BVH加速结构节点数组
static int g_numBVHNodes = 0;				//BVH节点总数
static ClippedTri* g_dClippedTris = nullptr;//裁剪后的三角形数组（每个原三角形最多2个）
static int g_maxClippedTris = 50000000;     //最大支持的裁剪后三角形数量（与原三角形相同）

//-------------------------- 核心工具函数 --------------------------
/**
 * 边函数：计算点相对于线段的位置
 * 返回值：有符号面积的两倍
 * 正数：点在线段右侧（顺时针方向）
 * 负数：点在线段左侧（逆时针方向）
 * 零：点在线段上
 */
__device__ __forceinline__
float edgeFunc(float ax, float ay, float bx, float by, float px, float py) {
	return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

/**
 * 近裁剪面裁剪：将三角形裁剪到 w >= zNear（即 z_ndc >= -1）
 * 这是解决"相机靠近模型消失"问题的核心函数
 * 原理：Sutherland-Hodgman算法，只对近裁剪面进行单次裁剪
 *
 * 输入：裁剪空间的三个顶点 (cx, cy, cz, cw)
 * 输出：裁剪后的顶点列表（最多2个三角形，6个顶点）
 * 返回值：输出的顶点数量（0=完全剔除，3=1个三角形，6=2个三角形）
 */
__device__ int clipTriangleToNearPlane(
	float cx0, float cy0, float cz0, float cw0,
	float cx1, float cy1, float cz1, float cw1,
	float cx2, float cy2, float cz2, float cw2,
	float outVerts[6][4]) //输出：裁剪后的顶点(x,y,z,w)
{
	//计算每个顶点到近裁剪面的距离
	//推导：z_ndc = cz/cw >= -1 → cz >= -cw → cz + cw >= 0
	//d > 0：顶点在近裁剪面之后（可见）
	//d <= 0：顶点在近裁剪面之前（不可见）
	float d0 = cz0 + cw0;
	float d1 = cz1 + cw1;
	float d2 = cz2 + cw2;

	//统计在近裁剪面之后的顶点数
	int insideCount = 0;
	if (d0 > 1e-6f) insideCount++;
	if (d1 > 1e-6f) insideCount++;
	if (d2 > 1e-6f) insideCount++;

	//情况1：所有顶点都在近裁剪面之后 → 不需要裁剪，直接返回原三角形
	if (insideCount == 3) {
		outVerts[0][0] = cx0; outVerts[0][1] = cy0; outVerts[0][2] = cz0; outVerts[0][3] = cw0;
		outVerts[1][0] = cx1; outVerts[1][1] = cy1; outVerts[1][2] = cz1; outVerts[1][3] = cw1;
		outVerts[2][0] = cx2; outVerts[2][1] = cy2; outVerts[2][2] = cz2; outVerts[2][3] = cw2;
		return 3;
	}

	//情况2：所有顶点都在近裁剪面之前 → 完全剔除，返回0
	if (insideCount == 0) {
		return 0;
	}

	//情况3：1个顶点在后面，2个在前面 → 裁剪成1个三角形
	if (insideCount == 1) {
		int i0, i1, i2;
		float d0_, d1_, d2_;
		float cx0_, cy0_, cz0_, cw0_;
		float cx1_, cy1_, cz1_, cw1_;
		float cx2_, cy2_, cz2_, cw2_;

		//找到在后面的那个顶点，统一为i0
		if (d0 > 1e-6f) {
			i0 = 0; i1 = 1; i2 = 2;
			d0_ = d0; d1_ = d1; d2_ = d2;
			cx0_ = cx0; cy0_ = cy0; cz0_ = cz0; cw0_ = cw0;
			cx1_ = cx1; cy1_ = cy1; cz1_ = cz1; cw1_ = cw1;
			cx2_ = cx2; cy2_ = cy2; cz2_ = cz2; cw2_ = cw2;
		}
		else if (d1 > 1e-6f) {
			i0 = 1; i1 = 2; i2 = 0;
			d0_ = d1; d1_ = d2; d2_ = d0;
			cx0_ = cx1; cy0_ = cy1; cz0_ = cz1; cw0_ = cw1;
			cx1_ = cx2; cy1_ = cy2; cz1_ = cz2; cw1_ = cw2;
			cx2_ = cx0; cy2_ = cy0; cz2_ = cz0; cw2_ = cw0;
		}
		else {
			i0 = 2; i1 = 0; i2 = 1;
			d0_ = d2; d1_ = d0; d2_ = d1;
			cx0_ = cx2; cy0_ = cy2; cz0_ = cz2; cw0_ = cw2;
			cx1_ = cx0; cy1_ = cy0; cz1_ = cz0; cw1_ = cw0;
			cx2_ = cx1; cy2_ = cy1; cz2_ = cz1; cw2_ = cw1;
		}

		//计算边 i0-i1 与近裁剪面的交点
		//线性插值：t = d0_ / (d0_ - d1_)
		float t1 = d0_ / (d0_ - d1_);
		float x1 = cx0_ + t1 * (cx1_ - cx0_);
		float y1 = cy0_ + t1 * (cy1_ - cy0_);
		float z1 = cz0_ + t1 * (cz1_ - cz0_);
		float w1 = cw0_ + t1 * (cw1_ - cw0_);

		//计算边 i0-i2 与近裁剪面的交点
		float t2 = d0_ / (d0_ - d2_);
		float x2 = cx0_ + t2 * (cx2_ - cx0_);
		float y2 = cy0_ + t2 * (cy2_ - cy0_);
		float z2 = cz0_ + t2 * (cz2_ - cz0_);
		float w2 = cw0_ + t2 * (cw2_ - cw0_);

		//输出裁剪后的三角形：[i0, 交点1, 交点2]
		outVerts[0][0] = cx0_; outVerts[0][1] = cy0_; outVerts[0][2] = cz0_; outVerts[0][3] = cw0_;
		outVerts[1][0] = x1;   outVerts[1][1] = y1;   outVerts[1][2] = z1;   outVerts[1][3] = w1;
		outVerts[2][0] = x2;   outVerts[2][1] = y2;   outVerts[2][2] = z2;   outVerts[2][3] = w2;
		return 3;
	}

	//情况4：2个顶点在后面，1个在前面 → 裁剪成2个三角形（4个顶点）
	else { //insideCount == 2
		int i0, i1, i2;
		float d0_, d1_, d2_;
		float cx0_, cy0_, cz0_, cw0_;
		float cx1_, cy1_, cz1_, cw1_;
		float cx2_, cy2_, cz2_, cw2_;

		//找到在前面的那个顶点，统一为i0
		if (d0 <= 1e-6f) {
			i0 = 0; i1 = 1; i2 = 2;
			d0_ = d0; d1_ = d1; d2_ = d2;
			cx0_ = cx0; cy0_ = cy0; cz0_ = cz0; cw0_ = cw0;
			cx1_ = cx1; cy1_ = cy1; cz1_ = cz1; cw1_ = cw1;
			cx2_ = cx2; cy2_ = cy2; cz2_ = cz2; cw2_ = cw2;
		}
		else if (d1 <= 1e-6f) {
			i0 = 1; i1 = 2; i2 = 0;
			d0_ = d1; d1_ = d2; d2_ = d0;
			cx0_ = cx1; cy0_ = cy1; cz0_ = cz1; cw0_ = cw1;
			cx1_ = cx2; cy1_ = cy2; cz1_ = cz2; cw1_ = cw2;
			cx2_ = cx0; cy2_ = cy0; cz2_ = cz0; cw2_ = cw0;
		}
		else {
			i0 = 2; i1 = 0; i2 = 1;
			d0_ = d2; d1_ = d0; d2_ = d1;
			cx0_ = cx2; cy0_ = cy2; cz0_ = cz2; cw0_ = cw2;
			cx1_ = cx0; cy1_ = cy0; cz1_ = cz0; cw1_ = cw0;
			cx2_ = cx1; cy2_ = cy1; cz2_ = cz1; cw2_ = cw1;
		}

		//计算边 i0-i1 与近裁剪面的交点
		float t1 = d1_ / (d1_ - d0_);
		float x1 = cx1_ + t1 * (cx0_ - cx1_);
		float y1 = cy1_ + t1 * (cy0_ - cy1_);
		float z1 = cz1_ + t1 * (cz0_ - cz1_);
		float w1 = cw1_ + t1 * (cw0_ - cw1_);

		//计算边 i0-i2 与近裁剪面的交点
		float t2 = d2_ / (d2_ - d0_);
		float x2 = cx2_ + t2 * (cx0_ - cx2_);
		float y2 = cy2_ + t2 * (cy0_ - cy2_);
		float z2 = cz2_ + t2 * (cz0_ - cz2_);
		float w2 = cw2_ + t2 * (cw0_ - cw2_);

		//输出第一个三角形：[i1, i2, 交点1]
		outVerts[0][0] = cx1_; outVerts[0][1] = cy1_; outVerts[0][2] = cz1_; outVerts[0][3] = cw1_;
		outVerts[1][0] = cx2_; outVerts[1][1] = cy2_; outVerts[1][2] = cz2_; outVerts[1][3] = cw2_;
		outVerts[2][0] = x1;   outVerts[2][1] = y1;   outVerts[2][2] = z1;   outVerts[2][3] = w1;

		//输出第二个三角形：[i2, 交点2, 交点1]
		outVerts[3][0] = cx2_; outVerts[3][1] = cy2_; outVerts[3][2] = cz2_; outVerts[3][3] = cw2_;
		outVerts[4][0] = x2;   outVerts[4][1] = y2;   outVerts[4][2] = z2;   outVerts[4][3] = w2;
		outVerts[5][0] = x1;   outVerts[5][1] = y1;   outVerts[5][2] = z1;   outVerts[5][3] = w1;
		return 6;
	}
}

//-------------------------- 预处理内核 --------------------------
/**
 * 三角形预处理内核：所有与像素无关的计算都在这里完成
 * 这是性能提升最大的优化点：将O(n²)的计算量降为O(n)
 *
 * 完成的工作：
 * 1. 计算裁剪空间坐标
 * 2. 近裁剪面裁剪（每个三角形只裁剪一次）
 * 3. 计算基础光照（漫反射+环境光，与像素位置无关）
 * 4. 计算精确的屏幕像素包围盒
 * 5. 生成裁剪后的三角形列表
 */
__global__ void computeTriCacheKernel(
	TriCached* triCache,
	ClippedTri* clippedTris,
	const GPUTriangle* tris,
	int numTris,
	RenderParams p)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x; //每个线程处理一个三角形
	if (t >= numTris) return; //超出范围的线程直接返回

	const GPUTriangle& tri = tris[t];
	TriCached& tc = triCache[t];

	float cx[3], cy[3], cz[3], cw[3]; //裁剪空间坐标
	int backCount = 0;                //在相机后方的顶点数
	bool needClip = false;            //是否需要进行近裁剪面裁剪

	//第一步：计算裁剪空间坐标
	for (int k = 0; k < 3; ++k) {
		float vx = tri.v[k].x, vy = tri.v[k].y, vz = tri.v[k].z;

		//MVP矩阵乘法：模型空间 → 裁剪空间
		cx[k] = p.mvp.m[0][0] * vx + p.mvp.m[1][0] * vy + p.mvp.m[2][0] * vz + p.mvp.m[3][0];
		cy[k] = p.mvp.m[0][1] * vx + p.mvp.m[1][1] * vy + p.mvp.m[2][1] * vz + p.mvp.m[3][1];
		cz[k] = p.mvp.m[0][2] * vx + p.mvp.m[1][2] * vy + p.mvp.m[2][2] * vz + p.mvp.m[3][2];
		cw[k] = p.mvp.m[0][3] * vx + p.mvp.m[1][3] * vy + p.mvp.m[2][3] * vz + p.mvp.m[3][3];

		//检查是否在相机后方（w <= 0）
		if (cw[k] <= 1e-6f) {
			backCount++;
			needClip = true;
		}
		//优化1：动态阈值调整
		//只有当w非常小时（<0.01）才需要裁剪，避免不必要的计算
		//这个值是经验值，平衡了性能和正确性
		else if (cw[k] < 0.01f) {
			needClip = true;
		}
	}

	//所有顶点都在相机后方，完全剔除这个三角形
	if (backCount == 3) {
		tc.minX = tc.maxX = tc.minY = tc.maxY = -2;
		tc.area = 0.f;
		tc.clippedTriCount = 0;
		return;
	}

	//第二步：基础光照计算（与裁剪无关，提前计算）
	//这部分计算与像素位置无关，每个三角形只需要计算一次
	Vec3 N = tri.normal;
	//计算原始三角形的面积和面朝向
	float area = edgeFunc(
		cx[0] / cw[0], cy[0] / cw[0],
		cx[1] / cw[1], cy[1] / cw[1],
		cx[2] / cw[2], cy[2] / cw[2]
	);
	bool flipped = (area < 0.f);
	if (flipped) N = -N; //背面使用相反的法线
	tc.flipped = flipped;
	tc.normal = N;

	//基础颜色转换：0-255 → 0.0-1.0
	float baseR = tri.color[0] / 255.f;
	float baseG = tri.color[1] / 255.f;
	float baseB = tri.color[2] / 255.f;
	//透明度设置
	tc.alpha = 1.0f;

	//主光源漫反射
	Vec3 L = p.lightDir;
	float keyDiff = fmaxf(N.dot(L), 0.f); //负数钳制为0（背面不接收光照）

	//补光漫反射（模拟环境光的方向性）
	Vec3 fillDir = { -0.4f, -0.3f, 0.6f };
	float fillDiff = fmaxf(N.dot(fillDir.norm()), 0.f);

	//半球环境光：朝上的面更亮，朝下的面更暗
	float hemi = 0.7f + 0.3f * N.y;

	//预计算基础色（高光除外，因为高光依赖视角/像素位置）
	tc.baseR = fminf(1.f,
		p.ambient.x * (0.8f + 0.2f * hemi) * baseR  //环境光
		+ keyDiff * p.lightColor.x * baseR * 0.85f  //主光源漫反射
		+ fillDiff * 0.35f * baseR                  //补光
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

	//第三步：近裁剪面裁剪（优化4：预处理阶段完成，每个三角形只裁剪一次）
	//这是解决"近距离帧率暴跌"的核心：将原本每个像素都要执行的裁剪，移到这里
	float clippedVerts[6][4];
	int numVerts = clipTriangleToNearPlane(
		cx[0], cy[0], cz[0], cw[0],
		cx[1], cy[1], cz[1], cw[1],
		cx[2], cy[2], cz[2], cw[2],
		clippedVerts
	);

	//裁剪后完全不可见，剔除
	if (numVerts == 0) {
		tc.minX = tc.maxX = tc.minY = tc.maxY = -2;
		tc.area = 0.f;
		tc.clippedTriCount = 0;
		return;
	}

	//第四步：生成裁剪后的三角形并计算精确包围盒（优化3）
	//替代了之前简单粗暴的全屏幕包围盒，大大减少了无效的三角形-像素测试
	tc.clippedTriStart = t * 2; //每个原三角形最多生成2个裁剪后三角形
	tc.clippedTriCount = numVerts / 3;

	float minX = 1e30f, maxX = -1e30f;
	float minY = 1e30f, maxY = -1e30f;

	for (int i = 0; i < tc.clippedTriCount; ++i) {
		ClippedTri& ct = clippedTris[tc.clippedTriStart + i];

		//透视除法得到NDC坐标（裁剪后执行，避免除以零）
		for (int k = 0; k < 3; ++k) {
			int vidx = i * 3 + k;
			ct.nx[k] = clippedVerts[vidx][0] / clippedVerts[vidx][3];
			ct.ny[k] = clippedVerts[vidx][1] / clippedVerts[vidx][3];
			ct.nz[k] = clippedVerts[vidx][2] / clippedVerts[vidx][3];
		}

		//计算裁剪后三角形的面积和面朝向
		float ctArea = edgeFunc(ct.nx[0], ct.ny[0], ct.nx[1], ct.ny[1], ct.nx[2], ct.ny[2]);
		ct.flipped = (ctArea < 0.f);
		ct.area = fabsf(ctArea);

		//更新整个原三角形的包围盒
		for (int k = 0; k < 3; ++k) {
			minX = fminf(minX, ct.nx[k]);
			maxX = fmaxf(maxX, ct.nx[k]);
			minY = fminf(minY, ct.ny[k]);
			maxY = fmaxf(maxY, ct.ny[k]);
		}
	}

	//第五步：计算精确的屏幕像素包围盒
	//NDC坐标(-1,-1)到(1,1) → 屏幕坐标(0,0)到(width-1, height-1)
	//Y轴翻转：NDC的(0,1)是屏幕顶部，(0,-1)是屏幕底部
	tc.minX = fmaxf(0, fminf(p.width - 1, (int)((minX + 1.0f) * 0.5f * p.width) - 1));
	tc.maxX = fmaxf(0, fminf(p.width - 1, (int)((maxX + 1.0f) * 0.5f * p.width) + 1));
	tc.minY = fmaxf(0, fminf(p.height - 1, (int)((1.0f - maxY) * 0.5f * p.height)));
	tc.maxY = fmaxf(0, fminf(p.height - 1, (int)((1.0f - minY) * 0.5f * p.height)));

	//保存原始NDC坐标（不需要裁剪的情况直接使用，避免重复计算）
	if (!needClip) {
		for (int k = 0; k < 3; ++k) {
			tc.nx[k] = cx[k] / cw[k];
			tc.ny[k] = cy[k] / cw[k];
			tc.nz[k] = cz[k] / cw[k];
		}
		tc.area = fabsf(area);
	}
}

//-------------------------- BVH加速内核 --------------------------
/**
 * 将BVH节点的模型空间AABB保守投影到屏幕像素空间
 * 用于快速剔除完全不可见的BVH节点
 */
__device__ bool nodeScreenBounds(const BVHNode& node, const Mat4& mvp,
	int width, int height,
	int& outMinX, int& outMaxX, int& outMinY, int& outMaxY)
{
	float minX = 1e30f, maxX = -1e30f;
	float minY = 1e30f, maxY = -1e30f;
	bool anyFront = false;
	bool nearPlaneIntersect = false;

	//AABB的8个角点坐标（模型空间）
	float cx[8] = { node.minX, node.maxX, node.minX, node.maxX,
					node.minX, node.maxX, node.minX, node.maxX };
	float cy[8] = { node.minY, node.minY, node.maxY, node.maxY,
					node.minY, node.minY, node.maxY, node.maxY };
	float cz[8] = { node.minZ, node.minZ, node.minZ, node.minZ,
					node.maxZ, node.maxZ, node.maxZ, node.maxZ };

	for (int i = 0; i < 8; ++i) {
		float x = cx[i], y = cy[i], z = cz[i];
		//MVP变换到裁剪空间
		float wx = mvp.m[0][0] * x + mvp.m[1][0] * y + mvp.m[2][0] * z + mvp.m[3][0];
		float wy = mvp.m[0][1] * x + mvp.m[1][1] * y + mvp.m[2][1] * z + mvp.m[3][1];
		float wz = mvp.m[0][2] * x + mvp.m[1][2] * y + mvp.m[2][2] * z + mvp.m[3][2];
		float ww = mvp.m[0][3] * x + mvp.m[1][3] * y + mvp.m[2][3] * z + mvp.m[3][3];

		if (ww <= 1e-6f) {
			nearPlaneIntersect = true;
			continue;
		}
		//优化1：与三角形预处理保持相同的阈值
		else if (ww < 0.01f) {
			nearPlaneIntersect = true;
		}
		anyFront = true;

		//透视除法 → NDC → 屏幕像素
		float ndcX = wx / ww;
		float ndcY = wy / ww;
		float sx = (ndcX + 1.0f) * 0.5f * width;
		float sy = (1.0f - ndcY) * 0.5f * height;

		minX = fminf(minX, sx); maxX = fmaxf(maxX, sx);
		minY = fminf(minY, sy); maxY = fmaxf(maxY, sy);
	}

	//所有角点都在相机后方，整个节点不可见
	if (!anyFront) return false;

	//优化3：如果与近裁剪面相交，保守地将包围盒扩展到整个屏幕
	//这是为了避免BVH节点被错误剔除，虽然会增加一些计算量，但比全屏幕包围盒好得多
	if (nearPlaneIntersect) {
		outMinX = 0;
		outMaxX = width - 1;
		outMinY = 0;
		outMaxY = height - 1;
	}
	else {
		//向外扩展1像素，确保保守性
		outMinX = fmaxf(0, (int)floorf(minX) - 1);
		outMaxX = fminf(width - 1, (int)ceilf(maxX) + 1);
		outMinY = fmaxf(0, (int)floorf(minY) - 1);
		outMaxY = fminf(height - 1, (int)ceilf(maxY) + 1);
	}
	return true;
}

//-------------------------- 分块构建内核 --------------------------
/**
 * 屏幕分块构建内核：为每个屏幕块生成覆盖它的三角形索引列表
 * 这是光栅化阶段的核心优化：将O(n)的三角形遍历降为O(k)，k是每个块的三角形数
 */
__global__ void buildTileKernel(
	TileData* tiles,
	const TriCached* triCache,
	int numTris,
	int tilesX, int tilesY,
	int width, int height,
	const Mat4 mvp,
	const BVHNode* bvhNodes,
	int numBVHNodes)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; //块的X坐标
	int ty = blockIdx.y * blockDim.y + threadIdx.y; //块的Y坐标
	if (tx >= tilesX || ty >= tilesY) return;

	int tileIdx = ty * tilesX + tx;
	TileData& tile = tiles[tileIdx];
	tile.triCount = 0;

	//当前块的像素范围
	int tileMinX = tx * TILE_SIZE;
	int tileMaxX = fminf((tx + 1) * TILE_SIZE - 1, width - 1);
	int tileMinY = ty * TILE_SIZE;
	int tileMaxY = fminf((ty + 1) * TILE_SIZE - 1, height - 1);

	//若BVH未上传（兼容旧逻辑），回退到暴力遍历
	if (!bvhNodes || numBVHNodes == 0) {
		for (int t = 0; t < numTris && tile.triCount < MAX_TRIS_PER_TILE; ++t) {
			const TriCached& tc = triCache[t];
			if (tc.clippedTriCount == 0) continue; //跳过已被剔除的三角形

			//检查三角形包围盒是否与当前块相交
			if (tc.maxX < tileMinX || tc.minX > tileMaxX ||
				tc.maxY < tileMinY || tc.minY > tileMaxY)
				continue;

			tile.triIdx[tile.triCount++] = t;
		}
		return;
	}

	//BVH栈遍历：只访问可能与当前块相交的节点
	int stack[64]; //显式栈，深度足够容纳2^64个节点（实际log2(N) < 25）
	int sp = 0;
	stack[sp++] = 0; //根节点索引固定为0

	while (sp > 0 && tile.triCount < MAX_TRIS_PER_TILE) {
		int nodeIdx = stack[--sp];
		const BVHNode& node = bvhNodes[nodeIdx];

		//1. 计算该节点在屏幕上的保守包围盒
		int nMinX, nMaxX, nMinY, nMaxY;
		if (!nodeScreenBounds(node, mvp, width, height, nMinX, nMaxX, nMinY, nMaxY))
			continue; //完全在视锥外，整棵子树剔除

		//2. 节点包围盒与当前块无交集？剪枝
		if (nMaxX < tileMinX || nMinX > tileMaxX ||
			nMaxY < tileMinY || nMinY > tileMaxY)
			continue;

		if (node.triCount > 0) {
			//3. 叶子节点：用精确的三角形包围盒二次过滤后加入块
			for (int i = 0; i < node.triCount && tile.triCount < MAX_TRIS_PER_TILE; ++i) {
				int t = node.triStart + i;
				if (t < 0 || t >= numTris) continue; //防御性检查
				const TriCached& tc = triCache[t];
				if (tc.clippedTriCount == 0) continue;

				if (tc.maxX < tileMinX || tc.minX > tileMaxX ||
					tc.maxY < tileMinY || tc.minY > tileMaxY)
					continue;
				tile.triIdx[tile.triCount++] = t;
			}
		}
		else {
			//4. 内部节点：将子节点压栈（先右后左，使左子树先被处理，提高缓存命中率）
			if (node.right >= 0) stack[sp++] = node.right;
			if (node.left >= 0) stack[sp++] = node.left;
		}
	}
}

//-------------------------- 主光栅化内核 --------------------------
/**
 * 主光栅化内核：每个线程处理一个像素
 * 优化后：使用预处理好的所有数据，不再进行任何裁剪或重复计算
 */
__global__
void rasterizeKernel(
	unsigned int* framebuf,
	float* depthbuf,
	const TriCached* triCache,
	const ClippedTri* clippedTris,
	const TileData* tiles,
	int tilesX,
	RenderParams p)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x; //像素X坐标
	int py = blockIdx.y * blockDim.y + threadIdx.y; //像素Y坐标
	if (px >= p.width || py >= p.height) return;    //超出屏幕范围的线程直接返回

	//找到当前像素属于哪个块
	int tx = px / TILE_SIZE;
	int ty = py / TILE_SIZE;
	int tileIdx = ty * tilesX + tx;
	const TileData& tile = tiles[tileIdx];
	if (tile.triCount == 0) return; //没有三角形覆盖的块直接返回

	int idx = py * p.width + px; //像素在帧缓冲中的线性索引
	//像素中心的NDC坐标（+0.5是为了采样像素中心）
	float ndcX = (px + 0.5f) / p.width * 2.f - 1.f;
	float ndcY = 1.f - (py + 0.5f) / p.height * 2.f;

	float bestZ = depthbuf[idx];			//当前最浅的深度值（初始为FLT_MAX）
	unsigned int bestColor = framebuf[idx];	//当前最浅像素的颜色

	//只遍历当前块中的三角形（这是分块优化的核心）
	for (int i = 0; i < tile.triCount; ++i) {
		int t = tile.triIdx[i];
		const TriCached& tc = triCache[t];
		if (tc.clippedTriCount == 0) continue;

		//包围盒快速剔除：如果像素不在三角形的包围盒内，直接跳过
		if (px < tc.minX || px > tc.maxX || py < tc.minY || py > tc.maxY)
			continue;

		//遍历裁剪后的所有三角形
		for (int ctIdx = 0; ctIdx < tc.clippedTriCount; ++ctIdx) {
			const ClippedTri& ct = clippedTris[tc.clippedTriStart + ctIdx];
			if (ct.area < 1e-6f) continue; //跳过退化三角形

			float nx0 = ct.nx[0], ny0 = ct.ny[0], nz0 = ct.nz[0];
			float nx1 = ct.nx[1], ny1 = ct.ny[1], nz1 = ct.nz[1];
			float nx2 = ct.nx[2], ny2 = ct.ny[2], nz2 = ct.nz[2];

			//处理面朝向：确保所有三角形都是顺时针绕序
			if (ct.flipped) {
				float tmp;
				tmp = nx0; nx0 = nx2; nx2 = tmp;
				tmp = ny0; ny0 = ny2; ny2 = tmp;
				tmp = nz0; nz0 = nz2; nz2 = tmp;
			}

			//计算重心坐标
			float w0 = edgeFunc(nx1, ny1, nx2, ny2, ndcX, ndcY) / ct.area;
			float w1 = edgeFunc(nx2, ny2, nx0, ny0, ndcX, ndcY) / ct.area;
			float w2 = 1.f - w0 - w1;

			//重心坐标有负数 → 像素在三角形外部，跳过
			if (w0 < -1e-6f || w1 < -1e-6f || w2 < -1e-6f) continue;

			//线框模式：只渲染三角形边缘
			if (p.wireframe) {
				float thresh = 0.025f; //线框宽度
				if (w0 > thresh && w1 > thresh && w2 > thresh) continue;
			}

			//深度测试：如果当前三角形比已渲染的更远，跳过
			float z = w0 * nz0 + w1 * nz1 + w2 * nz2;
			if (z >= bestZ) continue;
			bestZ = z; //更新最浅深度

			//Blinn-Phong光照计算（只有高光部分，基础色已预计算）
			Vec3 N = p.modelView.mulDir(tc.normal).norm();
			if (ct.flipped != tc.flipped) N = -N; //裁剪后面朝向可能改变

			Vec3 L = p.lightDir;
			Vec3 V = { 0.f,0.f,1.f }; //观察方向（看向屏幕外）
			Vec3 H = (L + V).norm();  //半程向量
			float spec = powf(fmaxf(N.dot(H), 0.f), p.specPower); //高光强度

			//合成最终颜色
			float r = fminf(1.f, tc.baseR + spec * 0.4f);
			float g = fminf(1.f, tc.baseG + spec * 0.4f);
			float b = fminf(1.f, tc.baseB + spec * 0.4f);

			//打包成BGRA格式（Windows DIB标准）
			bestColor = ((unsigned int)(tc.alpha * 255.0f) << 24)				//Alpha通道
				| ((unsigned int)(fminf(1.0f, b * tc.alpha) * 255.0f) << 16)	//蓝色分量
				| ((unsigned int)(fminf(1.0f, g * tc.alpha) * 255.0f) << 8)		//绿色分量
				|  (unsigned int)(fminf(1.0f, r * tc.alpha) * 255.0f);			//红色分量
		}
	}

	//写入最终结果
	depthbuf[idx] = bestZ;
	framebuf[idx] = bestColor;
}

//-------------------------- 清屏内核 --------------------------
__global__
void clearKernel(unsigned int* framebuf, float* depthbuf, int width, int height)
{
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	if (px >= width || py >= height) return;

	int idx = py * width + px;
	//垂直渐变背景：顶部暗灰，底部亮灰
	float t = (float)py / height;
	unsigned char bv = (unsigned char)(12 + t * 30);
	framebuf[idx] = 0xFF000000u | (bv << 16) | (bv << 8) | bv;
	//初始化深度缓冲区为最大值
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

	//释放可能已存在的旧缓冲区
	if (g_dFramebuf) cudaFree(g_dFramebuf);
	if (g_dDepthbuf) cudaFree(g_dDepthbuf);
	if (g_dTriCache) cudaFree(g_dTriCache);
	if (g_dTiles) cudaFree(g_dTiles);
	if (g_dClippedTris) cudaFree(g_dClippedTris);
	//分配新的GPU帧缓冲区和深度缓冲区
	cudaError_t err1 = cudaMalloc(&g_dFramebuf, sz * sizeof(unsigned int));  //帧缓冲（BGRA像素）
	cudaError_t err2 = cudaMalloc(&g_dDepthbuf, sz * sizeof(float));         //深度缓冲（浮点）
	cudaError_t err3 = cudaMalloc(&g_dTriCache, g_maxClippedTris / 2 * sizeof(TriCached)); //预处理数据缓冲
	cudaError_t err4 = cudaMalloc(&g_dTiles, tilesX * tilesY * sizeof(TileData));//分块内存
	cudaError_t err5 = cudaMalloc(&g_dClippedTris, g_maxClippedTris * sizeof(ClippedTri));//裁剪后三角形缓冲
	//检查内存分配是否成功
	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess ||
		err4 != cudaSuccess || err5 != cudaSuccess) return -1;
	return 0;  //成功
}

//三改：上传 BVH 节点数据到 GPU 显存（应在 cudaUploadTriangles 之后调用）
//nodes    - 主机端 BVH 节点数组指针（由 buildBVH 生成）
//numNodes - 节点总数
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

//将三角形数据上传到GPU显存（可多次调用以重新上传）
//参数：tris - 主机端三角形数组指针
//      numTris - 三角形数量
extern "C" CUDA_API
void cudaUploadTriangles(const GPUTriangle* tris, int numTris) {
	if (g_dTris) {
		cudaFree(g_dTris);     //释放GPU显存
		g_dTris = nullptr;     //重置指针
	}
	g_numTris = numTris;       //更新三角形计数

	//如果有新数据需要上传
	if (numTris > 0 && tris) {
		size_t bytes = (size_t)numTris * sizeof(GPUTriangle);  //计算字节数
		cudaError_t err = cudaMalloc(&g_dTris, bytes);         //分配GPU显存
		if (err != cudaSuccess) {
			//分配失败：重置状态
			g_dTris = nullptr;
			g_numTris = 0;
			return;
		}
		//将三角形数据从主机内存拷贝到设备显存
		cudaMemcpy(g_dTris, tris, bytes, cudaMemcpyHostToDevice);
	}
}

//渲染一帧：执行光栅化和光照计算
//参数：params - 渲染参数（MVP矩阵、光源等）
//      g_hFramebuf - 输出缓冲区指针（BGRA格式，至少 width*height*4 字节）
extern "C" CUDA_API
void cudaRender(const RenderParams& params, unsigned int* g_hFramebuf) {
	//检查GPU缓冲区和输出指针是否有效
	if (!g_dFramebuf || !g_dDepthbuf || !g_hFramebuf) return;
	if (!g_dTris || !g_dTriCache || !g_dTiles || !g_dClippedTris) return;
	//配置CUDA线程网格（每个线程块16×16线程）
	dim3 block(16, 16);
	//算需要线程块数，向上取整
	dim3 grid((params.width + block.x - 1) / block.x,
		(params.height + block.y - 1) / block.y);
	//第一步：清屏
	clearKernel << <grid, block >> > (g_dFramebuf, g_dDepthbuf, params.width, params.height);

	//第二步：GPU预计算每个三角形的NDC坐标+屏幕包围盒
	int triBlocks = (params.numTris + 255) / 256;
	computeTriCacheKernel << <triBlocks, 256 >> > (
		g_dTriCache,
		g_dClippedTris,
		g_dTris,
		params.numTris,
		params);

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
		params.mvp,         //三改：传入 MVP 矩阵供 BVH 投影使用
		g_dBVH,				//三改:传入设备端 BVH 指针
		g_numBVHNodes);		//三改:传入节点数量

	//第四步：光栅化三角形
	rasterizeKernel << <grid, block >> > (
		g_dFramebuf,
		g_dDepthbuf,
		g_dTriCache,
		g_dClippedTris,
		g_dTiles,
		tilesX,
		params);
	cudaDeviceSynchronize();
	//第五步：显存结果拷贝回内存
	cudaMemcpy(g_hFramebuf, g_dFramebuf,
		(size_t)params.width * params.height * sizeof(unsigned int),
		cudaMemcpyDeviceToHost);
}

//释放所有GPU资源：在卸载DLL之前调用
extern "C" CUDA_API
void cudaShutdown(void)
{
	//释放帧缓冲区
	if (g_dFramebuf) {
		cudaFree(g_dFramebuf);
		g_dFramebuf = nullptr;
	}
	//释放深度缓冲区
	if (g_dDepthbuf) {
		cudaFree(g_dDepthbuf);
		g_dDepthbuf = nullptr;
	}
	//释放三角形数据
	if (g_dTris) {
		cudaFree(g_dTris);
		g_dTris = nullptr;
	}
	//释放预处理缓存
	if (g_dTriCache) {
		cudaFree(g_dTriCache);
		g_dTriCache = nullptr;
	}
	//释放分块
	if (g_dTiles) {
		cudaFree(g_dTiles);
		g_dTiles = nullptr;
	}
	//释放裁剪后三角形缓存
	if (g_dClippedTris) {
		cudaFree(g_dClippedTris);
		g_dClippedTris = nullptr;
	}
	//释放 BVH 节点
	if (g_dBVH) {
		cudaFree(g_dBVH);
		g_dBVH = nullptr;
	}
	g_numBVHNodes = 0;
	g_numTris = 0;  //重置三角形计数
}