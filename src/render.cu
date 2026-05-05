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

//边函数，屏幕向下看，判断点在线的左or右(正顺右，负逆左)，返回有符号面积（正向屏上，负向屏下）
__device__ __forceinline__
float edgeFunc(float ax, float ay, float bx, float by, float px, float py) {
	return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

//主光栅化，对于每个像素
__global__
void rasterizeKernel(
	unsigned int* framebuf,	//
	float* depthbuf,
	const GPUTriangle* tris,//三角形数组（只读）
	RenderParams p			//
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;//屏幕x坐标像素
	int py = blockIdx.y * blockDim.y + threadIdx.y;//屏幕y坐标像素

	if (px >= p.width || py >= p.height)return;//跳过显示区域外的多余像素的线程处理
	int idx = py * p.width + px;//像素索引

	//+0.5选像素中心，然后归一化到NDC空间
	float ndcX= (px + 0.5f) / p.width * 2.f - 1.f;
	float ndcY = 1.f - (py + 0.5f) / p.height * 2.f;

	float bestZ = depthbuf[idx];			//最前边的深度值，开始是FLT_MAX
	unsigned int bestColor = framebuf[idx];	//最前边的物体的颜色，开始是clear画的灰色

	//定一个像素，挨个看三角形
	for (int t = 0; t < p.numTris; ++t) {
		const GPUTriangle& tri = tris[t];
		//顶点0 1 2的MVP变换
		float cx0 = p.mvp.m[0][0] * tri.v[0].x + p.mvp.m[1][0] * tri.v[0].y + p.mvp.m[2][0] * tri.v[0].z + p.mvp.m[3][0];
		float cy0 = p.mvp.m[0][1] * tri.v[0].x + p.mvp.m[1][1] * tri.v[0].y + p.mvp.m[2][1] * tri.v[0].z + p.mvp.m[3][1];
		float cz0 = p.mvp.m[0][2] * tri.v[0].x + p.mvp.m[1][2] * tri.v[0].y + p.mvp.m[2][2] * tri.v[0].z + p.mvp.m[3][2];
		float cw0 = p.mvp.m[0][3] * tri.v[0].x + p.mvp.m[1][3] * tri.v[0].y + p.mvp.m[2][3] * tri.v[0].z + p.mvp.m[3][3];
		float cx1 = p.mvp.m[0][0] * tri.v[1].x + p.mvp.m[1][0] * tri.v[1].y + p.mvp.m[2][0] * tri.v[1].z + p.mvp.m[3][0];
		float cy1 = p.mvp.m[0][1] * tri.v[1].x + p.mvp.m[1][1] * tri.v[1].y + p.mvp.m[2][1] * tri.v[1].z + p.mvp.m[3][1];
		float cz1 = p.mvp.m[0][2] * tri.v[1].x + p.mvp.m[1][2] * tri.v[1].y + p.mvp.m[2][2] * tri.v[1].z + p.mvp.m[3][2];
		float cw1 = p.mvp.m[0][3] * tri.v[1].x + p.mvp.m[1][3] * tri.v[1].y + p.mvp.m[2][3] * tri.v[1].z + p.mvp.m[3][3];
		float cx2 = p.mvp.m[0][0] * tri.v[2].x + p.mvp.m[1][0] * tri.v[2].y + p.mvp.m[2][0] * tri.v[2].z + p.mvp.m[3][0];
		float cy2 = p.mvp.m[0][1] * tri.v[2].x + p.mvp.m[1][1] * tri.v[2].y + p.mvp.m[2][1] * tri.v[2].z + p.mvp.m[3][1];
		float cz2 = p.mvp.m[0][2] * tri.v[2].x + p.mvp.m[1][2] * tri.v[2].y + p.mvp.m[2][2] * tri.v[2].z + p.mvp.m[3][2];
		float cw2 = p.mvp.m[0][3] * tri.v[2].x + p.mvp.m[1][3] * tri.v[2].y + p.mvp.m[2][3] * tri.v[2].z + p.mvp.m[3][3];
		//近平面裁剪：如果任何顶点的w<=0，说明在摄像机后方
		if (cw0 <= 0.f || cw1 <= 0.f || cw2 <= 0.f) continue;
		//透视除法
		float nx0 = cx0 / cw0, ny0 = cy0 / cw0, nz0 = cz0 / cw0; //点0
		float nx1 = cx1 / cw1, ny1 = cy1 / cw1, nz1 = cz1 / cw1; //点1
		float nx2 = cx2 / cw2, ny2 = cy2 / cw2, nz2 = cz2 / cw2; //点2
		//背面剔除
		float area = edgeFunc(nx0, ny0, nx1, ny1, nx2, ny2);
		if (area == 0.f) continue;  // 退化三角形（面积为零，跳过）
		bool flipped = (area < 0.f);//看到背面
		if (flipped) {
			area = -area;  // 取绝对值
			// 交换顶点0和顶点2的标签，翻转面朝向，保证点都顺时针，便于用w>=0判断在内部
			float tmp;
			tmp = nx0; nx0 = nx2; nx2 = tmp;  // 交换X坐标
			tmp = ny0; ny0 = ny2; ny2 = tmp;  // 交换Y坐标
			tmp = nz0; nz0 = nz2; nz2 = tmp;  // 交换Z坐标
		}
		//粗：包围盒剔除（CUDA的最值函数求包围盒）
		float minX = fminf(nx0, fminf(nx1, nx2));
		float maxX = fmaxf(nx0, fmaxf(nx1, nx2));
		float minY = fminf(ny0, fminf(ny1, ny2));
		float maxY = fmaxf(ny0, fmaxf(ny1, ny2));
		if (ndcX < minX || ndcX > maxX || ndcY < minY || ndcY > maxY) continue;//包围盒外，跳过该三角形
		//细：重心坐标计算比较（权重和为1，而且面积占的越大离相应点越近，w也能反映距离）
		float w0 = edgeFunc(nx1, ny1, nx2, ny2, ndcX, ndcY) / area;	//顶点 V0 的权重 = P 到对边 V1-V2 的距离比例
		float w1 = edgeFunc(nx2, ny2, nx0, ny0, ndcX, ndcY) / area;	//顶点 V1 的权重 = P 到对边 V2-V0 的距离比例
		float w2 = 1.f - w0 - w1;									//剩下的权重
		if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;				//有负，则在三角形外部，跳过该三角形

		//线框模式
		if (p.wireframe) {
			float thresh = 0.025f;	//线框宽度
			if (w0 > thresh && w1 > thresh && w2 > thresh) continue;//离框太远不渲染
		}
		//深度测试
		float z = w0 * nz0 + w1 * nz1 + w2 * nz2;	//三点深度换成三角形深度
		if (z >= bestZ) continue;  //被更近的遮挡，不渲染
		bestZ = z;                  // 更新最佳深度值


		//Blinn_Phong光照
		//提前把0-255转换成0.0-1.0
		float baseR = tri.color[0] / 255.f;  //基础红色分量
		float baseG = tri.color[1] / 255.f;  //基础绿色分量
		float baseB = tri.color[2] / 255.f;  //基础蓝色分量
		
		Vec3 N = tri.normal;				 //三角形面法线
		if (flipped) { N = -N; }			 //背面光照：使用相反法线

		//漫反射
		Vec3 L = p.lightDir;				 //光照方向,指向光源
		float keyDiff = fmaxf(N.dot(L), 0.f);//漫反射系数（值越大越亮，范围0-1，负数被0.f钳制到0）

		//高光
		Vec3 V = { 0.f,0.f,1.f };	//观察方向，看向屏幕外
		Vec3 H = (L + V).norm();	//半程向量
		float spec = powf(fmaxf(N.dot(H), 0.f), p.specPower);  // 高光强度（指数控制高光集中度）

		//补光
		Vec3 fillDir = { -0.4f, -0.3f, 0.6f };					// 补光方向
		float fillDiff = fmaxf(N.dot(fillDir.norm()), 0.f);		// 补光漫反射系数

		//半球环境光,朝上亮，朝下暗
		float hemi = 0.7f + 0.3f * N.y;

		//合成光照，3通道分别合成
		float r = fminf(1.f,								//光照为1或者
			p.ambient.x * (0.8f + 0.2f * hemi) * baseR		//环境光
			+ keyDiff * p.lightColor.x * baseR * 0.85f		//漫反射主光
			+ fillDiff * 0.35 * baseR						//补光
			+ spec * 0.4f									//高光
		);
		float g = fminf(1.f,
			p.ambient.y * (0.8f + 0.2f * hemi) * baseG
			+ keyDiff * p.lightColor.y * baseG * 0.85f
			+ fillDiff * 0.3f * baseG
			+ spec * 0.4f
		);
		float b = fminf(1.f,
			p.ambient.z * (0.8f + 0.2f * hemi) * baseB
			+ keyDiff * p.lightColor.z * baseB * 0.85f
			+ fillDiff * 0.4f * baseB
			+ spec * 0.4f
		);

		//打包成BGRA（Windows DIB标准）,浮点RGB的0-1转为32位二进制,用unsigned int位运算
		bestColor=0xFF000000u					// 透明度FF决定不透明（最高位）
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

	// 释放可能已存在的旧缓冲区
	if (g_dFramebuf) cudaFree(g_dFramebuf);
	if (g_dDepthbuf) cudaFree(g_dDepthbuf);

	// 分配新的GPU帧缓冲区和深度缓冲区
	cudaError_t err1 = cudaMalloc(&g_dFramebuf, sz * sizeof(unsigned int));  // 帧缓冲（BGRA像素）
	cudaError_t err2 = cudaMalloc(&g_dDepthbuf, sz * sizeof(float));         // 深度缓冲（浮点）

	// 检查内存分配是否成功
	if (err1 != cudaSuccess || err2 != cudaSuccess) return -1;
	return 0;  // 成功
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
	// 配置CUDA线程网格（每个线程块16×16线程）
	dim3 block(16, 16);
	//算需要线程块数，向上取整
	dim3 grid((params.width + block.x - 1) / block.x,
		(params.height + block.y - 1) / block.y);
	//第一步：清屏
	clearKernel << <grid, block >> > (g_dFramebuf, g_dDepthbuf, params.width, params.height);
	cudaDeviceSynchronize();  // 等待清屏完成
	//第二步：光栅化三角形
	rasterizeKernel << <grid, block >> > (g_dFramebuf, g_dDepthbuf,
		g_dTris ? g_dTris : nullptr,  // 如果没有三角形，传nullptr
		params);
	//第三步：显存结果拷贝回内存
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
	g_numTris = 0;  // 重置三角形计数
}