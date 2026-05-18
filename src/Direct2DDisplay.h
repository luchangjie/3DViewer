//===================================================================
// Direct2DDisplay.h - 使用 Direct2D 高效显示像素缓冲区
// 功能：将 BGRA 格式的像素数组显示到 Windows 窗口
//===================================================================
#pragma once
#include <windows.h>
#include <d2d1.h>           // Direct2D 头文件

#pragma comment(lib, "d2d1.lib")  // 链接 Direct2D 库

// Direct2D 显示器类
// 用于将 CPU/GPU 渲染的像素数据快速显示到窗口
class Direct2DDisplay {
private:
    ID2D1Factory* pFactory = nullptr;           // Direct2D 工厂对象（用于创建资源）
    ID2D1HwndRenderTarget* pRT = nullptr;       // 窗口渲染目标（绑定到特定窗口）
    int width = 0, height = 0;                  // 显示区域的宽高（像素）
    bool valid = false;                         // 是否初始化成功

public:
    // 析构函数：自动释放资源
    ~Direct2DDisplay() { cleanup(); }

    // 释放所有 Direct2D 资源
    void cleanup() {
        if (pRT) { pRT->Release(); pRT = nullptr; }        // 释放渲染目标
        if (pFactory) { pFactory->Release(); pFactory = nullptr; } // 释放工厂
        valid = false;
    }

    // 初始化 Direct2D
    // hwnd - 目标窗口句柄
    // w, h - 显示区域宽高
    // 返回值：成功返回 true，失败返回 false
    bool init(HWND hwnd, int w, int h) {
        cleanup();                          // 先清理旧资源
        width = w;
        height = h;

        // 创建 Direct2D 工厂（单线程模式，因为只有一个渲染线程）
        HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);
        if (FAILED(hr)) return false;

        // 创建窗口渲染目标
        // 像素格式：BGRA 32位，预乘 Alpha 模式
        hr = pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(
                D2D1_RENDER_TARGET_TYPE_DEFAULT,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED)
            ),
            D2D1::HwndRenderTargetProperties(hwnd, D2D1::SizeU(w, h)),
            &pRT
        );
        if (FAILED(hr)) return false;

        valid = true;
        return true;
    }

    // 绘制像素数据到窗口
    // pixels - BGRA 格式的像素数组（大小为 width * height）
    void draw(unsigned int* pixels) {
        if (!valid || !pRT || !pixels) return;

        // 配置 Bitmap 属性：BGRA 格式，96 DPI
        D2D1_BITMAP_PROPERTIES props = {};
        props.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
        props.pixelFormat.alphaMode = D2D1_ALPHA_MODE_PREMULTIPLIED;
        props.dpiX = 96.0f;
        props.dpiY = 96.0f;

        // 从像素数据创建 Direct2D Bitmap
        ID2D1Bitmap* pBitmap = nullptr;
        D2D1_SIZE_U size = { (UINT32)width, (UINT32)height };

        HRESULT hr = pRT->CreateBitmap(
            size,                           // Bitmap 尺寸
            pixels,                         // 像素数据指针
            width * 4,                      // 行跨度（每行字节数 = 宽 × 4字节）
            props,                          // 像素格式属性
            &pBitmap                        // 输出 Bitmap 对象
        );
        if (FAILED(hr) || !pBitmap) return;

        // 开始绘制
        pRT->BeginDraw();

        // 清空背景（深灰，作为未覆盖区域的底色）
        pRT->Clear(D2D1::ColorF(0.1f, 0.1f, 0.12f,1.0f));

        // 将 Bitmap 绘制到窗口的整个客户区
        D2D1_RECT_F dest = { 0, 0, (float)width, (float)height };
        pRT->DrawBitmap(pBitmap, &dest);    // 目标矩形 = 窗口客户区大小

        // 结束绘制（检查是否需要重建渲染目标）
        hr = pRT->EndDraw();
        pBitmap->Release();                 // 立即释放临时 Bitmap

        // 如果因窗口大小改变等原因丢失设备，标记为无效
        // 调用方需要重新 init()
        if (hr == D2DERR_RECREATE_TARGET) {
            valid = false;
            pRT->Release();
            pRT = nullptr;
        }
    }
};