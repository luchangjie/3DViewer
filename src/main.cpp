// ==================================================================
//  main.cpp - Win32窗口 + 消息循环 + 摄像机控制
//  构建方式：仅使用MSVC（无需CUDA头文件，运行时加载render_kernel.dll）
//  操作说明：
//    鼠标左键拖拽   - 旋转模型
//    鼠标滚轮       - 缩放（推近/拉远）
//    W键            - 切换线框/实体显示模式
//    O键            - 打开OBJ文件对话框
//    R键            - 重置摄像机位置
//    ESC键          - 退出程序
// ==================================================================
#define WIN32_LEAN_AND_MEAN		//精简Windows头文件，排除不常用的API声明
#define NOMINMAX				//禁止Windows.h定义min/max宏（避免与std::min/max冲突）
#include<windows.h>				//windowsAPI
#include<commdlg.h>				//使程序可以打开文件
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cfloat>
#include<string>
#include"render.h"

int loadOBJ(const char* path, GPUTriangle*& triangles);	        //在obj_loader.cpp
int loadVoxelObj(const char* path, GPUTriangle*& triangles);    //在voxel_loader.cpp

typedef int (*cudaInit_t)(int, int, char*, int);
typedef void (*cudaUploadTriangles_t)(const GPUTriangle*, int);  
typedef void (*cudaRender_t)(const RenderParams&, unsigned int*);
typedef void (*cudaShutdown_t)(void);                            

//全局指针变量（通过GetProcAddress获取DLL中函数地址），分别指向dll的4个函数
static cudaInit_t				p_cudaInit = nullptr;
static cudaUploadTriangles_t	p_cudaUploadTriangles = nullptr;
static cudaRender_t				p_cudaRender = nullptr;
static cudaShutdown_t			 p_cudaShutdown = nullptr;

//全局常量
static const int	WIN_W = 900;	//窗口宽
static const int	WIN_H = 700;	//窗口高
static const float	FOV_Y = 60.f * 3.14159265f / 180.f;	//垂直视场角（弧度制）
static const float	Z_NEAR = 0.1f;	//近裁剪面
static const float  Z_FAR = 100.f;	// 远裁剪面距离

static HMODULE g_hKernel = nullptr;	//windows句柄类型

//加载DLL并获取所有CUDA函数指针，初始化
static bool initRenderer() {
	g_hKernel = LoadLibraryA("render_kernel.dll");	//Windows给dll的句柄
    if (!g_hKernel) {
        // 加载失败：显示错误消息框
        MessageBoxA(nullptr,
            "Cannot load render_kernel.dll!\nPlease run build_all.bat first.",
            "3D Viewer", MB_ICONERROR);
        return false;
    }
    //调用GetProcAddress获取dll里的函数，图省事用了宏，##拼，#单独
#define LOAD_SYM(name) p_##name = (name##_t)GetProcAddress(g_hKernel, #name)
    LOAD_SYM(cudaInit);            
    LOAD_SYM(cudaUploadTriangles);
    LOAD_SYM(cudaRender);          
    LOAD_SYM(cudaShutdown);        
#undef LOAD_SYM
    //检查函数是否加载成功
    if (!p_cudaInit || !p_cudaRender || !p_cudaShutdown) {
        MessageBoxA(nullptr,
            "render_kernel.dll: missing required exports!\nPlease rebuild.",
            "BlackFrame 3D Viewer", MB_ICONERROR);
        return false;
    }

    // 调用CUDA初始化函数
    char gpuName[256] = {};              // 用于接收GPU名称的缓冲区
    if (p_cudaInit(WIN_W, WIN_H, gpuName, sizeof(gpuName)) != 0) {
        // 初始化失败（无NVIDIA GPU或其他错误）
        MessageBoxA(nullptr,
            "CUDA initialization failed!\nNo NVIDIA GPU detected.",
            "BlackFrame 3D Viewer", MB_ICONERROR);
        return false;
    }
    // 打印检测到的GPU设备名称
    printf("[CUDA] Device: %s\n", gpuName);
    return true;
}
//-------------------------------------------------------------------
//全局状态变量
static unsigned int*    g_hFramebuf = nullptr;  // 指向DIB像素数据的指针
static GPUTriangle*     g_hTris = nullptr;      // 三角形数组（主机端）
static int              g_numTris = 0;          // 三角形数量
static float     g_rotX = 0.3f;                 // 摄像机绕X轴旋转角度（弧度）
static float     g_rotY = 0.5f;                 // 摄相机绕Y轴旋转角度（弧度）
static float     g_zoom = 4.5f;                 // 摄像机距离（缩放级别）
static bool      g_dragging = false;            // 鼠标是否正在拖拽（左键按住）
static int       g_lastMX = 0, g_lastMY = 0;    // 上一帧的鼠标位置（用于计算增量）
static bool      g_wireframe = false;           //是否线框模式
static bool      g_voxelMode = false;           //是否体素模式
static std::string g_currentModelPath;          //记住当前模型路径
// Windows资源句柄
static HWND         g_hwnd = nullptr;       // 主窗口句柄 Handle to a Window
static HDC          g_hdc = nullptr;        // 窗口画布句柄 Handle to a Device Context
static HBITMAP      g_hbmp = nullptr;       // DIB位图句柄 Handle to a BITMAP
static HDC          g_memDC = nullptr;      // 内存画布句柄（双缓冲）
static void*        g_dibBits = nullptr;    // DIB位图像素数据指针
//-------------------------------------------------------------------
//打开文件 对话框
static std::string openFileDialog(HWND owner) {
    OPENFILENAMEA ofn = {};                 //文件对话框结构体
    ofn.lStructSize = sizeof(ofn);          //Windows要求设置结构体大小
    ofn.hwndOwner = owner;                  //父窗口是owner
    ofn.lpstrFilter = "OBJ Files (*.obj)\0*.obj\0All Files\0*.*\0";
    char buf[MAX_PATH] = {};                //定个260字节的数组
    ofn.lpstrFile = buf;                    //指向buf来填路径
    ofn.nMaxFile = MAX_PATH;                //缓冲区大小
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;  //文件必须存在，路径必须存在
    return GetOpenFileNameA(&ofn) ? std::string(buf) : "";  //打开文件选择窗口（判断条件：“打开”则返回true，return字符串）
}
//-------------------------------------------------------------------
//加载obj并上传至GPU
static void loadModel(const std::string& path) {
    // 释放旧的三角形数据
    delete[] g_hTris;
    g_hTris = nullptr;
    g_numTris = 0;

    g_currentModelPath = path;      //记录路径
    // 如果路径非空，解析OBJ文件(两种方式)
    if (!path.empty()) {
        if(g_voxelMode)
            g_numTris = loadVoxelObj(path.c_str(), g_hTris);
        else
            g_numTris = loadOBJ(path.c_str(), g_hTris); //string转为char*作实参
    }
    // 将三角形数据上传到GPU显存
    if (p_cudaUploadTriangles)
        p_cudaUploadTriangles(g_hTris, g_numTris);
    // 重置摄像机到默认视角
    g_rotX = 0.3f; g_rotY = 0.5f; g_zoom = 4.5f;
}
//-------------------------------------------------------------------
//创建DIB段（GPU渲染结果搬到CPU存的地）
static void createDIB(HWND hwnd, int w, int h) {
    // 清理旧的GDI资源
    if (g_memDC)  DeleteDC(g_memDC);                    // 删除内存的画布
    if (g_hbmp)   DeleteObject(g_hbmp);                 // 删除位图对象
    if (g_hdc)    ReleaseDC(hwnd, g_hdc);               // 释放窗口的画布

    g_hdc = GetDC(hwnd);                                // 凭句柄hwnd获取窗口画布 Device Context
    g_memDC = CreateCompatibleDC(g_hdc);                // 创建内存画布（双缓冲）

    //配置位图数据
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);    // 结构体大小
    bmi.bmiHeader.biWidth = w;                          // 位图宽度
    bmi.bmiHeader.biHeight = -h;                        // 负值表示从上到下存储（原点在左上角）
    bmi.bmiHeader.biPlanes = 1;                         // 颜色平面数（历史遗留字段，固定为1）
    bmi.bmiHeader.biBitCount = 32;                      // 每像素32位（BGRA）
    bmi.bmiHeader.biCompression = BI_RGB;               // 无压缩

    g_hbmp = CreateDIBSection(g_hdc, &bmi, DIB_RGB_COLORS, &g_dibBits, nullptr, 0);//创建位图对象，g_dibBits有了指向
    SelectObject(g_memDC, g_hbmp);                      //位图选入内存画布
    g_hFramebuf = (unsigned int*)g_dibBits;             //像素数据指针转为无符号整型指针
}

//-------------------------------------------------------------------
//集成好的逐帧渲染
static void renderFrame() {
    // 检查帧缓冲和渲染函数是否可用
    if (!g_hFramebuf || !p_cudaRender) return;
    // 构建模型矩阵（先绕Y轴旋转，再绕X轴旋转）
    Mat4 model = Mat4::rotateX(g_rotX) * Mat4::rotateY(g_rotY);
    //摄像机位置
    Vec3 eye = { 0.f,0.f,g_zoom }; 
    //构建视图矩阵
    Mat4 view = Mat4::lookAt(eye, { 0,0,0 }, { 0,1,0 });
    // 计算宽高比
    float aspect = (float)WIN_W / WIN_H;
    // 构建透视投影矩阵
    Mat4 proj = Mat4::perspective(FOV_Y, aspect, Z_NEAR, Z_FAR);
    //填充渲染参数结构体
    RenderParams rp;
    rp.mvp = proj * view * model;               //模型-视图-投影 矩阵：模型空间到裁剪空间
    rp.modelView = view * model;                //模型-视图矩阵：用于将法线变换到视图空间进行光照计算
    rp.lightDir = Vec3(0.6f, 0.8f, -0.5f).norm();//世界空间光照方向，指向光源
    rp.lightColor = { 1.0f, 0.98f, 0.95f };     //光源颜色
    rp.ambient = { 0.3f, 0.3f, 0.3f };          //环境光颜色
    rp.width = WIN_W;                           // 帧缓冲宽度（像素）
    rp.height = WIN_H;                          // 帧缓冲高度（像素）
    rp.numTris = g_numTris;                     // 三角形总数     
    rp.wireframe = g_wireframe ? 1 : 0;         //非0为绘制线框模式
    rp.specPower = 64.f;                        //高光指数（像镜面程度）

    // 调用CUDA渲染（GPU计算 + 拷贝回主机内存）
    p_cudaRender(rp, g_hFramebuf);
    // 将渲染结果从内存DC拷贝到窗口DC（双缓冲显示）
    BitBlt(g_hdc, 0, 0, WIN_W, WIN_H, g_memDC, 0, 0, SRCCOPY);
    // FPS 显示
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
        "3DViewer  |  Tris: %d  |  FPS: %.1f  |  Mode: %s  |  [W]Wire [V]Voxel [O]Open [R]Reset",
        g_numTris,fps, modeStr);
    SetWindowTextA(g_hwnd, title);
}

//-------------------------------------------------------------------
//Win32窗口消息回调函数（其实就是对各自操作的反应）
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_DESTROY: {                  // 窗口被销毁
        PostQuitMessage(0);             // 发送退出消息，终止消息循环
        return 0;
    }
    case WM_KEYDOWN: {                  // 键盘按下
        if (wp == VK_ESCAPE) {          // ESC键：退出程序
            PostQuitMessage(0);
        }
        else if (wp == 'W') {           // W键：切换线框/实体模式
            g_wireframe = !g_wireframe;
            InvalidateRect(hwnd, nullptr, FALSE);  // 标记窗口需要重绘
        }
        else if (wp == 'V') {           // V键：切换体素模式
            g_voxelMode = !g_voxelMode;
            loadModel(g_currentModelPath);       
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (wp == 'R') {           // R键：重置摄像机
            g_rotX = 0.3f; g_rotY = 0.5f; g_zoom = 4.5f;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (wp == 'O') {           // O键：打开OBJ文件
            std::string path = openFileDialog(hwnd);
            if (!path.empty()) loadModel(path);
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    }
    case WM_LBUTTONDOWN: {            // 鼠标左键按下
        g_dragging = true;            // 开始拖拽
        g_lastMX = LOWORD(lp);        // 记录当前鼠标X坐标（低16位）
        g_lastMY = HIWORD(lp);        // 记录当前鼠标Y坐标（高16位）
        SetCapture(hwnd);             // 捕获鼠标（即使鼠标移出窗口也能接收消息）
        return 0;
    }
    case WM_MOUSEMOVE: {               // 鼠标移动
        if (g_dragging) {             // 仅在拖拽状态下处理
            int mx = LOWORD(lp), my = HIWORD(lp);  // 当前鼠标位置
            float dx = (mx - g_lastMX) * 0.008f;   // 水平移动增量（转换为旋转量）
            float dy = (my - g_lastMY) * 0.008f;   // 垂直移动增量
            g_rotY += dx;             // 水平拖拽=绕Y轴旋转
            g_rotX += dy;             // 垂直拖拽=绕X轴旋转
            // 限制X轴旋转角度（防止模型翻转）
            if (g_rotX > 1.5f) g_rotX = 1.5f;
            if (g_rotX < -1.5f) g_rotX = -1.5f;
            g_lastMX = mx; g_lastMY = my;  // 更新上一帧鼠标位置
            InvalidateRect(hwnd, nullptr, FALSE);  // 请求重绘
        }
        return 0;
    }
    case WM_LBUTTONUP: {              // 鼠标左键释放
        g_dragging = false;           // 停止拖拽
        ReleaseCapture();             // 释放鼠标捕获
        return 0;
    }
    case WM_MOUSEWHEEL: {             // 鼠标滚轮
        int delta = GET_WHEEL_DELTA_WPARAM(wp);  // 获取滚轮增量（正值=向前滚）
        g_zoom -= delta * 0.005f;     // 调整缩放距离，因为模型居中且摄像机看向中，故等价缩放了
        // 限制缩放范围
        if (g_zoom < 0.5f) g_zoom = 0.5f;  // 最近距离
        if (g_zoom > 20.f) g_zoom = 20.f;  // 最远距离
        InvalidateRect(hwnd, nullptr, FALSE);  //请求重绘
        return 0;
    }
    case WM_PAINT: {                  // 窗口绘制消息，响应重绘请求，一次处理积压的请求，节省资源
        PAINTSTRUCT ps;
        BeginPaint(hwnd, &ps);        // 开始绘制（获取绘制区域）
        renderFrame();                // 执行渲染
        EndPaint(hwnd, &ps);          // 结束绘制
        return 0;
    }
    case WM_ERASEBKGND:               // 擦除背景消息（重绘时Windows自己想擦，这样会闪屏）
        return 1;                     // 返回非零值表示已处理（防止闪烁）
    }
    // 调用默认窗口过程处理其他消息（没定义的右键、移动窗口什么的）
    return DefWindowProcA(hwnd, msg, wp, lp);
}

//-------------------------------------------------------------------
//WinMain(windows的main函数)
int WINAPI WinMain(HINSTANCE hInst, //程序身份标记
    HINSTANCE,                      //过时的东西，永远为空
    LPSTR lpCmdLine,                //命令行参数，暂时没用
    int nShow)                      //窗口初始显示方式（最大化，最小化，正常）
{                     
    // 分配控制台窗口（用于显示printf的调试信息）
    AllocConsole();
    freopen("CONOUT$", "w", stdout);   // 重定向标准输出到控制台
    freopen("CONOUT$", "w", stderr);   // 重定向标准错误到控制台

    //初始化渲染器，加载DLL + 初始化CUDA
    if (!initRenderer()) return 1;     // 初始化失败则退出

    //注册窗口类
    WNDCLASSEXA wc = {};
    wc.cbSize = sizeof(wc);                   // 结构体大小
    wc.style = CS_HREDRAW | CS_VREDRAW;      // 窗口大小改变时重绘
    wc.lpfnWndProc = WndProc;                      // 窗口消息处理函数
    wc.hInstance = hInst;                        // 应用程序实例句柄
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW); // 默认箭头光标
    wc.lpszClassName = "3DViewer";                 // 窗口类名
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION); // 默认应用图标
    RegisterClassExA(&wc);                            // 注册窗口类

    // 计算窗口实际大小（考虑标题栏和边框）
    RECT rc = { 0, 0, WIN_W, WIN_H };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    int ww = rc.right - rc.left;     // 实际窗口宽度
    int wh = rc.bottom - rc.top;     // 实际窗口高度

    // 创建窗口
    g_hwnd = CreateWindowExA(
        0, "3DViewer",                          // 窗口类名
        "3D Viewer",                   // 窗口标题
        WS_OVERLAPPEDWINDOW,                      // 窗口样式（可调整大小）
        CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,     // 位置和大小
        nullptr, nullptr, hInst, nullptr);
    if (!g_hwnd) { fprintf(stderr, "CreateWindow failed\n"); return 1; }

    // 显示窗口
    ShowWindow(g_hwnd, nShow);
    UpdateWindow(g_hwnd);

    //---------------------------------------------------------------
    //创建DIB缓冲区（帧缓冲）
    createDIB(g_hwnd, WIN_W, WIN_H);

    //默认加载cube.obj模型
    {
        char exeDir[MAX_PATH] = {};                // 可执行文件路径缓冲区
        GetModuleFileNameA(nullptr, exeDir, MAX_PATH);  // 获取exe完整路径
        char* lastSlash = strrchr(exeDir, '\\');   // 查找最后一个反斜杠
        if (lastSlash) *(lastSlash + 1) = '\0';      // 截断到目录路径
        std::string defaultModel = std::string(exeDir) + "models\\cube.obj";  // 拼接默认模型路径
        loadModel(defaultModel);                   // 加载默认模型
    }

    //消息循环
    MSG msg = {};
    while (GetMessageA(&msg, nullptr, 0, 0)) {     // 获取消息（阻塞直到有消息）
        TranslateMessage(&msg);                     // 翻译虚拟键消息
        DispatchMessageA(&msg);                     // 分发消息到窗口过程
    }

    // -- 释放资源 --------------------------------------------------------------
    if (p_cudaShutdown) p_cudaShutdown();          // 关闭CUDA释放显存
    delete[] g_hTris;                               // 释放三角形数据
    if (g_memDC)  DeleteDC(g_memDC);               // 删除内存设备上下文
    if (g_hbmp)   DeleteObject(g_hbmp);            // 删除位图对象
    if (g_hdc && g_hwnd) ReleaseDC(g_hwnd, g_hdc); // 释放窗口设备上下文
    if (g_hKernel) FreeLibrary(g_hKernel);          // 卸载DLL

    return (int)msg.wParam;                        // 返回退出代码
}
