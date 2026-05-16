//===================================================================
// render_engine.cpp - CUDA 渲染器管理、BVH 构建、模型加载与帧渲染
//===================================================================
#include "app_core.h"

//-------------------------------------------------------------------
// 初始化 CUDA 渲染器：加载 DLL 并获取函数指针
//-------------------------------------------------------------------
bool initRenderer() {
	g_hKernel = LoadLibraryA("render_kernel.dll");	// Windows 加载 DLL 的句柄
	if (!g_hKernel) {
		// 加载失败，显示错误信息框
		MessageBoxA(nullptr,
			"Cannot load render_kernel.dll!\nPlease run build_all.bat first.",
			"3D Viewer", MB_ICONERROR);
		return false;
	}
	// 使用 GetProcAddress 获取 DLL 中的函数地址，节省内存（## 拼接 # 字符串）
#define LOAD_SYM(name) p_##name = (name##_t)GetProcAddress(g_hKernel, #name)
	LOAD_SYM(cudaInit);
	LOAD_SYM(cudaUploadBVH);
	LOAD_SYM(cudaUploadTriangles);
	LOAD_SYM(cudaRender);
	LOAD_SYM(cudaShutdown);
#undef LOAD_SYM
	// 检查必要函数是否成功导出
	if (!p_cudaInit || !p_cudaRender || !p_cudaShutdown) {
		MessageBoxA(nullptr,
			"render_kernel.dll: missing required exports!\nPlease rebuild.",
			"BlackFrame 3D Viewer", MB_ICONERROR);
		return false;
	}

	// 执行 CUDA 初始化
	char gpuName[256] = {};				  // 用于接收 GPU 名称的缓冲区
	if (p_cudaInit(WIN_W, WIN_H, gpuName, sizeof(gpuName)) != 0) {
		// 初始化失败，无 NVIDIA GPU 或驱动问题
		MessageBoxA(nullptr,
			"CUDA initialization failed!\nNo NVIDIA GPU detected.",
			"BlackFrame 3D Viewer", MB_ICONERROR);
		return false;
	}
	// 打印检测到的 GPU 设备名称
	printf("[CUDA] Device: %s\n", gpuName);
	return true;
}

//-------------------------------------------------------------------
// BVH 构建：内部结构体与递归函数
//-------------------------------------------------------------------
// BVH 构建辅助结构：记录每个原始三角形的 AABB 和中心点
struct BVHBuildTri {
	int  origIdx;			// 原始三角形索引（构建后用于重排）
	Vec3 center;			// 包围盒中心
	Vec3 bmin, bmax;		// 包围盒范围
};

// 递归构建 BVH，返回节点在 outNodes 中的索引
// 注意：构建完成后 buildTris 会被原地重排，使得叶子节点的三角形连续
static int buildBVHRecursive(std::vector<BVHBuildTri>& buildTris,
	int start, int end,
	std::vector<BVHNode>& outNodes)
{
	// 关键：先记录当前节点在 outNodes 中的索引（emplace_back 可能使引用失效）
	int nodeIdx = (int)outNodes.size();
	outNodes.emplace_back();			// 创建新节点，注意后续不能持有引用

	// 计算当前范围 [start, end) 内所有三角形的联合 AABB
	Vec3 lo(FLT_MAX, FLT_MAX, FLT_MAX);
	Vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (int i = start; i < end; ++i) {
		lo.x = fminf(lo.x, buildTris[i].bmin.x);
		lo.y = fminf(lo.y, buildTris[i].bmin.y);
		lo.z = fminf(lo.z, buildTris[i].bmin.z);
		hi.x = fmaxf(hi.x, buildTris[i].bmax.x);
		hi.y = fmaxf(hi.y, buildTris[i].bmax.y);
		hi.z = fmaxf(hi.z, buildTris[i].bmax.z);
	}
	// 填充当前节点的 AABB（注意：这里不能通过引用 outNodes[nodeIdx] 访问）
	outNodes[nodeIdx].minX = lo.x;
	outNodes[nodeIdx].minY = lo.y;
	outNodes[nodeIdx].minZ = lo.z;
	outNodes[nodeIdx].maxX = hi.x;
	outNodes[nodeIdx].maxY = hi.y;
	outNodes[nodeIdx].maxZ = hi.z;

	int count = end - start;
	const int LEAF_MAX = 8;

	if (count <= LEAF_MAX) {
		// 叶子节点
		outNodes[nodeIdx].left = -1;
		outNodes[nodeIdx].right = -1;
		outNodes[nodeIdx].triStart = start;
		outNodes[nodeIdx].triCount = count;
		return nodeIdx;
	}

	// 内部节点：按最长轴分割
	Vec3 extent = hi - lo;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

	int mid = start + count / 2;
	std::nth_element(buildTris.begin() + start,
		buildTris.begin() + mid,
		buildTris.begin() + end,
		[axis](const BVHBuildTri& a, const BVHBuildTri& b) {
			float av = (axis == 0) ? a.center.x : (axis == 1) ? a.center.y : a.center.z;
			float bv = (axis == 0) ? b.center.x : (axis == 1) ? b.center.y : b.center.z;
			return av < bv;
		});

	// 关键：递归前先记录；递归内部会再次 emplace_back，nodeIdx 指向的旧位置仍然有效
	int leftIdx = buildBVHRecursive(buildTris, start, mid, outNodes);
	int rightIdx = buildBVHRecursive(buildTris, mid, end, outNodes);

	outNodes[nodeIdx].left = leftIdx;
	outNodes[nodeIdx].right = rightIdx;
	outNodes[nodeIdx].triStart = -1;
	outNodes[nodeIdx].triCount = 0;
	return nodeIdx;
}

// 构建接口：为三角形数组构建 BVH，并原地重排三角形顺序
// 重排后 BVH 叶子节点的 triStart 直接对应 tris 数组下标，无需额外映射
void buildBVH(GPUTriangle* tris, int numTris, std::vector<BVHNode>& outNodes)
{
	outNodes.clear();
	if (numTris <= 0) return;

	// 1. 为每个原始三角形计算 AABB 和中心点
	std::vector<BVHBuildTri> buildTris(numTris);
	for (int i = 0; i < numTris; ++i) {
		const GPUTriangle& tri = tris[i];
		Vec3 lo = { fminf(tri.v[0].x, fminf(tri.v[1].x, tri.v[2].x)),
					fminf(tri.v[0].y, fminf(tri.v[1].y, tri.v[2].y)),
					fminf(tri.v[0].z, fminf(tri.v[1].z, tri.v[2].z)) };
		Vec3 hi = { fmaxf(tri.v[0].x, fmaxf(tri.v[1].x, tri.v[2].x)),
					fmaxf(tri.v[0].y, fmaxf(tri.v[1].y, tri.v[2].y)),
					fmaxf(tri.v[0].z, fmaxf(tri.v[1].z, tri.v[2].z)) };
		buildTris[i].origIdx = i;
		buildTris[i].bmin = lo;
		buildTris[i].bmax = hi;
		buildTris[i].center = (lo + hi) * 0.5f;
	}

	// 2. 递归构建 BVH，节点写入 outNodes，同时确定重排顺序
	buildBVHRecursive(buildTris, 0, numTris, outNodes);

	// 3. 按 buildTris 的新顺序原地重排原始三角形数组
	//    使得 BVH 叶子节点记录的 triStart/triCount 直接指向连续内存
	std::vector<GPUTriangle> temp(numTris);
	for (int i = 0; i < numTris; ++i) {
		temp[i] = tris[buildTris[i].origIdx];
	}
	for (int i = 0; i < numTris; ++i) {
		tris[i] = temp[i];
	}
}

//-------------------------------------------------------------------
// 加载 OBJ 并上传到 GPU
//-------------------------------------------------------------------
void loadModel(const std::string& path) {
	// 释放旧的三角形数组
	delete[] g_hTris;
	g_hTris = nullptr;
	g_numTris = 0;
	g_hBVH.clear();					// 清空旧 BVH
	g_currentModelPath = path;		// 记录路径
	// 如果路径非空，加载 OBJ 文件（支持普通或体素模式）
	if (!path.empty()) {
		if (g_voxelMode)
			g_numTris = loadVoxelObj(path.c_str(), g_hTris);
		else
			g_numTris = loadOBJ(path.c_str(), g_hTris); // string 转为 char* 实现
	}
	// 如果加载成功，原地构建 BVH 并上传 GPU
	if (g_numTris > 0) {
		buildBVH(g_hTris, g_numTris, g_hBVH);
		g_numBVHNodes = (int)g_hBVH.size();
	}
	// 将三角形数据上传到 GPU 显存
	if (p_cudaUploadTriangles)
		p_cudaUploadTriangles(g_hTris, g_numTris);
	// 上传 BVH 节点到显存
	if (p_cudaUploadBVH && g_numBVHNodes > 0)
		p_cudaUploadBVH(g_hBVH.data(), g_numBVHNodes);
	// 重置相机到默认视角
	g_rotX = 0.3f; g_rotY = 0.5f; g_zoom = 4.0f;
}

//-------------------------------------------------------------------
// 生成并渲染一帧
//-------------------------------------------------------------------
void renderFrame() {
	// 如果帧缓冲或渲染函数未就绪，直接返回
	if (!g_hFramebuf || !p_cudaRender) return;
	// 计算模型矩阵：先绕 Y 轴旋转，再绕 X 轴旋转
	Mat4 model = Mat4::rotateX(g_rotX) * Mat4::rotateY(g_rotY);
	// 相机位置
	Vec3 eye = { g_panX, g_panY, g_zoom };
	Vec3 center = { g_panX, g_panY, 0.f };	  // 观察目标（与相机保持同一平面）
	// 视图矩阵
	Mat4 view = Mat4::lookAt(eye, center, { 0,1,0 });
	// 窗口宽高比
	float aspect = (float)WIN_W / WIN_H;
	// 透视投影矩阵
	Mat4 proj = Mat4::perspective(FOV_Y, aspect, Z_NEAR, Z_FAR);
	// 填充渲染参数结构体
	RenderParams rp;
	rp.mvp = proj * view * model;				   // 模型-视图-投影 矩阵（模型空间到裁剪空间）
	rp.modelView = view * model;				   // 模型-视图矩阵（用于法线变换）
	rp.lightDir = Vec3(0.6f, 0.8f, -0.5f).norm();// 模型空间主光源方向（指向光源）
	rp.lightColor = { 1.0f, 0.98f, 0.95f };		   // 光源颜色
	rp.ambient = { 0.3f, 0.3f, 0.3f };			   // 环境光颜色
	rp.width = WIN_W;							   // 帧缓冲宽度（像素）
	rp.height = WIN_H;							   // 帧缓冲高度（像素）
	rp.numTris = g_numTris;						   // 三角形数量
	rp.wireframe = g_wireframe ? 1 : 0;			   // 非 0 为线框模式
	rp.specPower = 64.f;						   // 高光指数（控制高光锐度）

	// 调用 CUDA 渲染（GPU 光栅化 + 光照计算），结果写入 g_hFramebuf
	p_cudaRender(rp, g_hFramebuf);
	// 将渲染结果从内存位图绘制到窗口 DC（通过 Direct2D）
	g_display.draw(g_hFramebuf);
	// FPS 计算
	static DWORD lastTime = 0;
	static int frameCount = 0;
	static float fps = 0.0f;
	frameCount++;
	DWORD curTime = GetTickCount();
	if (curTime - lastTime >= 500) {
		fps = frameCount * 1000.0f / (curTime - lastTime);
		frameCount = 0;
		lastTime = curTime;
	}
	// 更新窗口标题栏信息
	char title[128];
	const char* modeStr = g_voxelMode ? "Voxel" : (g_wireframe ? "Wire" : "Solid");
	snprintf(title, sizeof(title),
		"3DViewer | Tris: %d | FPS: %.1f | Mode: %s | [W]Wire [V]Voxel [G]Get voxel obj [O]Open [R]Reset",
		g_numTris, fps, modeStr);
	SetWindowTextA(g_hwnd, title);
}
