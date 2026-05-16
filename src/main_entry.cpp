//===================================================================
// main_entry.cpp - Win32 入口：WinMain + 默认模型加载 + 消息循环
//===================================================================
#include "app_core.h"

//-------------------------------------------------------------------
// 全局变量定义（单一定义点，其他文件通过 app_core.h 的 extern 引用）
//-------------------------------------------------------------------

// CUDA DLL
cudaInit_t				p_cudaInit = nullptr;
cudaUploadTriangles_t	p_cudaUploadTriangles = nullptr;
cudaRender_t			p_cudaRender = nullptr;
cudaShutdown_t			p_cudaShutdown = nullptr;
cudaUploadBVH_t         p_cudaUploadBVH = nullptr;
HMODULE g_hKernel = nullptr;

// 全局常量
const int	WIN_W = 900;
const int	WIN_H = 700;
const float	FOV_Y = 60.f * 3.14159265f / 180.f;
const float	Z_NEAR = 0.01f;
const float Z_FAR = 100.f;

// BVH
std::vector<BVHNode> g_hBVH;
int                  g_numBVHNodes = 0;

// 帧缓冲与模型数据
unsigned int* g_hFramebuf = nullptr;
GPUTriangle* g_hTris = nullptr;
int              g_numTris = 0;

// 交互状态
float     g_rotX = 0.3f;
float     g_rotY = 0.5f;
float     g_zoom = 4.0f;
bool      g_dragging = false;
int       g_lastMX = 0, g_lastMY = 0;
bool      g_panning = false;
float     g_panX = 0.f, g_panY = 0.f;
bool      g_wireframe = false;
bool      g_voxelMode = false;
std::string g_currentModelPath;

// Win32 资源
HWND         g_hwnd = nullptr;
Direct2DDisplay g_display;

//-------------------------------------------------------------------
// WinMain（windows 的 main 函数）
//-------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hInst, // 当前应用程序实例句柄
	HINSTANCE,						  // 16 位遗留参数，始终为 NULL
	LPSTR lpCmdLine,				  // 命令行参数（本例未使用）
	int nShow)						  // 窗口初始显示方式（最大化、最小化、正常）
{
	// COM 初始化
	CoInitialize(nullptr);

	// 创建独立控制台窗口，用于显示 printf 的调试信息
	AllocConsole();
	freopen("CONOUT$", "w", stdout);	 // 重定向标准输出到控制台
	freopen("CONOUT$", "w", stderr);	 // 重定向标准错误到控制台

	// 初始化渲染器（加载 DLL + 初始化 CUDA）
	if (!initRenderer()) return 1;		 // 初始化失败则退出

	// 注册窗口类
	WNDCLASSEXA wc = {};
	wc.cbSize = sizeof(wc);					   // 结构体大小
	wc.style = CS_HREDRAW | CS_VREDRAW;		   // 窗口大小改变时重绘
	wc.lpfnWndProc = WndProc;					   // 设置消息回调函数
	wc.hInstance = hInst;						   // 应用程序实例句柄
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW); // 默认箭头光标
	wc.lpszClassName = "3DViewer";				   // 窗口类名
	wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION); // 默认应用程序图标
	RegisterClassExA(&wc);							   // 注册窗口类

	// 计算窗口实际大小（包含标题栏和边框尺寸）
	RECT rc = { 0, 0, WIN_W, WIN_H };
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
	int ww = rc.right - rc.left;	 // 实际窗口宽度
	int wh = rc.bottom - rc.top;	 // 实际窗口高度

	// 创建主窗口
	g_hwnd = CreateWindowExA(
		0, "3DViewer",							  // 窗口类名
		"3D Viewer",							  // 窗口标题
		WS_OVERLAPPEDWINDOW,					  // 窗口样式（可调整大小）
		CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,   // 位置和大小
		nullptr, nullptr, hInst, nullptr);
	if (!g_hwnd) { fprintf(stderr, "CreateWindow failed\n"); return 1; }

	// 显示并更新窗口
	ShowWindow(g_hwnd, nShow);
	UpdateWindow(g_hwnd);

	// 初始化显示（Direct2D）
	initDisplay(g_hwnd, WIN_W, WIN_H);

	// 默认加载一个内嵌的立方体模型
	{
		// 内嵌立方体 OBJ 数据（C 风格字符串，只读数据段）
		static const char* embeddedPoint =
			"v  -1.0  -1.0  -1.0\n"
			"v  -1.0   1.0  -1.0\n"
			"v   1.0   1.0  -1.0\n"
			"v   1.0  -1.0  -1.0\n"
			"v  -1.0  -1.0   1.0\n"
			"v  -1.0   1.0   1.0\n"
			"v   1.0   1.0   1.0\n"
			"v   1.0  -1.0   1.0\n"
			"f 1 2 3\n"
			"f 1 3 4\n"
			"f 5 6 2\n"
			"f 5 2 1\n"
			"f 6 7 3\n"
			"f 6 3 2\n"
			"f 7 8 4\n"
			"f 7 4 3\n"
			"f 8 5 1\n"
			"f 8 1 4\n"
			"f 5 8 7\n"
			"f 5 7 6\n";
		// OBJ 格式要求顶点索引从 1 开始，面索引直接对应

		char tempPath[MAX_PATH];	 // 系统临时目录路径
		char tempFile[MAX_PATH];	 // 临时文件完整路径

		GetTempPathA(sizeof(tempPath), tempPath);		// 获取系统临时目录路径
		GetTempFileNameA(tempPath, "3dv", 0, tempFile); // 在临时目录生成唯一文件名
		// 前缀 "3dv"，后缀 .tmp，例如：
		// C:\...\Temp\3dv1234.tmp
		std::ofstream ofs(tempFile);
		if (ofs) {
			ofs << embeddedPoint;
			ofs.close();
			loadModel(tempFile);
		}
		DeleteFileA(tempFile);
	}

	// 消息循环
	MSG msg = {};
	while (GetMessageA(&msg, nullptr, 0, 0)) {	  // 获取消息，阻塞直到有消息
		TranslateMessage(&msg);					  // 翻译键盘消息
		DispatchMessageA(&msg);					  // 分发消息到窗口回调
	}

	// -- 释放资源 --------------------------------------------------------------
	if (p_cudaShutdown) p_cudaShutdown();		  // 关闭 CUDA 释放显存
	delete[] g_hTris;								  // 释放主机端三角形数组
	delete[] g_hFramebuf;
	if (g_hKernel) FreeLibrary(g_hKernel);		  // 卸载 DLL

	CoUninitialize();
	return (int)msg.wParam;						  // 返回退出代码
}
