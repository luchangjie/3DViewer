//===================================================================
// window_ui.cpp - Win32 窗口回调、文件对话框与 Direct2D 显示初始化
//===================================================================
#include "app_core.h"

//-------------------------------------------------------------------
// 打开文件对话框
//-------------------------------------------------------------------
std::string openFileDialog(HWND owner) {
	OPENFILENAMEA ofn = {};					 // 文件对话框结构体
	ofn.lStructSize = sizeof(ofn);			 // Windows 要求设置结构体大小
	ofn.hwndOwner = owner;					 // 设置父窗口 owner
	ofn.lpstrFilter = "OBJ Files (*.obj)\0*.obj\0All Files\0*.*\0";
	char buf[MAX_PATH] = {};				 // 缓冲 260 字节的文件路径
	ofn.lpstrFile = buf;					 // 指向 buf 作为路径缓冲区
	ofn.nMaxFile = MAX_PATH;				 // 缓冲区最大长度
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;  // 文件必须存在，路径必须存在
	return GetOpenFileNameA(&ofn) ? std::string(buf) : "";  // 弹出文件选择窗口，若用户点击“打开”则返回 true，返回字符串
}

//-------------------------------------------------------------------
// 初始化 Direct2D 显示
//-------------------------------------------------------------------
void initDisplay(HWND hwnd, int w, int h) {
	g_display.init(hwnd, w, h);
	g_hFramebuf = new unsigned int[w * h];
}

//-------------------------------------------------------------------
// Win32 窗口消息回调函数：处理鼠标、键盘、重绘等
//-------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
	case WM_DESTROY: {					  // 窗口被关闭
		PostQuitMessage(0);				  // 发送退出消息，结束消息循环
		return 0;
	}
	case WM_KEYDOWN: {					  // 键盘按下
		if (wp == VK_ESCAPE) {			  // ESC 键退出程序
			PostQuitMessage(0);
		}
		else if (wp == 'W') {			  // W 键切换线框/实体模式
			g_wireframe = !g_wireframe;
			InvalidateRect(hwnd, nullptr, FALSE);  // 标记窗口需要重绘
		}
		else if (wp == 'V') {			  // V 键切换体素模式
			g_voxelMode = !g_voxelMode;
			loadModel(g_currentModelPath);
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		else if (wp == 'R') {			  // R 键重置相机
			g_rotX = 0.3f; g_rotY = 0.5f; g_zoom = 4.0f;
			g_panX = 0.f; g_panY = 0.f;
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		else if (wp == 'O') {			  // O 键打开 OBJ 文件
			g_voxelMode = false;
			std::string path = openFileDialog(hwnd);
			if (!path.empty()) loadModel(path);
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		else if (wp == 'G' && g_voxelMode && g_numTris > 0) {   // G 键导出体素化 obj 文件
			OPENFILENAMEA ofn = {};
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hwnd;
			ofn.lpstrFilter = "OBJ Files (*.obj)\0*.obj\0All Files\0*.*\0";
			char buf[MAX_PATH] = "voxelized.obj";
			ofn.lpstrFile = buf;
			ofn.nMaxFile = MAX_PATH;
			ofn.Flags = OFN_OVERWRITEPROMPT;

			if (GetSaveFileNameA(&ofn)) {
				// 调用导出函数
				if (exportVoxelObj(buf, g_hTris, g_numTris)) {
					MessageBoxA(hwnd, "体素化 OBJ 导出成功！", "导出成功", MB_OK | MB_ICONINFORMATION);
				}
				else {
					MessageBoxA(hwnd, "导出失败：无法写入文件。", "错误", MB_OK | MB_ICONERROR);
				}
			}
		}
		return 0;
	}
	case WM_LBUTTONDOWN: {				  // 鼠标左键按下
		g_dragging = true;				  // 开始拖拽旋转
		g_lastMX = LOWORD(lp);			  // 记录当前鼠标 X 坐标（低 16 位）
		g_lastMY = HIWORD(lp);			  // 记录当前鼠标 Y 坐标（高 16 位）
		SetCapture(hwnd);				  // 捕获鼠标（即使移出窗口也能接收消息）
		return 0;
	}
	case WM_RBUTTONDOWN: {				  // 鼠标右键按下
		g_panning = true;
		g_lastMX = LOWORD(lp);
		g_lastMY = HIWORD(lp);
		SetCapture(hwnd);
		return 0;
	}
	case WM_MOUSEMOVE: {
		if (g_dragging) {			  // 左键拖拽
			int mx = LOWORD(lp), my = HIWORD(lp);  // 当前鼠标位置
			float dx = (mx - g_lastMX) * 0.008f;   // 水平移动量转换为旋转角度
			float dy = (my - g_lastMY) * 0.008f;   // 垂直移动量
			g_rotY += dx;				  // 水平拖拽 = 绕 Y 轴旋转
			g_rotX += dy;				  // 垂直拖拽 = 绕 X 轴旋转
			// 限制 X 轴旋转角度，防止模型翻转过头
			if (g_rotX > 1.5f) g_rotX = 1.5f;
			if (g_rotX < -1.5f) g_rotX = -1.5f;
			g_lastMX = mx; g_lastMY = my;  // 更新上一帧鼠标位置
			InvalidateRect(hwnd, nullptr, FALSE);  // 请求重绘
		}
		else if (g_panning) {		  // 右键拖拽平移
			int mx = LOWORD(lp), my = HIWORD(lp);
			float dx = (mx - g_lastMX) * 0.003f;  // 灵敏度
			float dy = (my - g_lastMY) * 0.003f;
			g_panX -= dx;
			g_panY += dy;
			g_lastMX = mx; g_lastMY = my;
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		return 0;
	}
	case WM_LBUTTONUP: {				  // 鼠标左键释放
		g_dragging = false;			  // 停止拖拽
		ReleaseCapture();			  // 释放鼠标捕获
		return 0;
	}
	case WM_RBUTTONUP: {				  // 鼠标右键释放
		g_panning = false;
		ReleaseCapture();
		return 0;
	}
	case WM_MOUSEWHEEL: {				  // 鼠标滚轮
		int delta = GET_WHEEL_DELTA_WPARAM(wp);  // 获取滚轮增量（正值向上）
		float sensitivity;
		if (g_zoom < 2.0f) {
			// 近距离
			sensitivity = 0.001f;
		}
		else if (g_zoom < 6.0f) {
			// 中距离
			sensitivity = 0.002f;
		}
		else {
			// 远距离
			sensitivity = 0.005f;
		}
		g_zoom -= delta * sensitivity;
		// 限制缩放范围
		if (g_zoom < 0.2f) g_zoom = 0.2f;  // 最近距离
		if (g_zoom > 16.f) g_zoom = 16.f;  // 最远距离
		InvalidateRect(hwnd, nullptr, FALSE);  // 请求重绘
		return 0;
	}
	case WM_PAINT: {					  // 窗口绘制消息：响应重绘请求（第一次创建、被遮挡后恢复等）
		PAINTSTRUCT ps;
		BeginPaint(hwnd, &ps);		  // 开始绘制，获取绘制设备上下文
		renderFrame();				  // 执行渲染
		EndPaint(hwnd, &ps);		  // 结束绘制
		return 0;
	}
	case WM_ERASEBKGND:				  // 擦除背景消息：重绘时 Windows 自动发送，这里直接拦截
		return 1;					  // 返回非零值表示已处理背景，防止闪烁
	}
	// 其他默认窗口消息（如鼠标移动未拖拽等）交给默认处理
	return DefWindowProcA(hwnd, msg, wp, lp);
}
