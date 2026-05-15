#pragma once
#include <windows.h>
#include <d2d1.h>

#pragma comment(lib, "d2d1.lib")

class Direct2DDisplay {
private:
    ID2D1Factory* pFactory = nullptr;
    ID2D1HwndRenderTarget* pRT = nullptr;
    int width = 0, height = 0;
    bool valid = false;

public:
    ~Direct2DDisplay() { cleanup(); }

    void cleanup() {
        if (pRT) { pRT->Release(); pRT = nullptr; }
        if (pFactory) { pFactory->Release(); pFactory = nullptr; }
        valid = false;
    }

    bool init(HWND hwnd, int w, int h) {
        cleanup();
        width = w;
        height = h;

        HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);
        if (FAILED(hr)) return false;

        RECT rc = { 0, 0, w, h };
        hr = pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(
                D2D1_RENDER_TARGET_TYPE_DEFAULT,
                D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE)
            ),
            D2D1::HwndRenderTargetProperties(hwnd, D2D1::SizeU(w, h)),
            &pRT
        );
        if (FAILED(hr)) return false;

        valid = true;
        return true;
    }

    void draw(unsigned int* pixels) {
        if (!valid || !pRT || !pixels) return;

        D2D1_BITMAP_PROPERTIES props = {};
        props.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
        props.pixelFormat.alphaMode = D2D1_ALPHA_MODE_IGNORE;
        props.dpiX = 96.0f;
        props.dpiY = 96.0f;

        ID2D1Bitmap* pBitmap = nullptr;
        D2D1_SIZE_U size = { (UINT32)width, (UINT32)height };

        HRESULT hr = pRT->CreateBitmap(
            size,
            pixels,
            width * 4,      // stride
            props,
            &pBitmap
        );
        if (FAILED(hr) || !pBitmap) return;

        pRT->BeginDraw();
        pRT->Clear(D2D1::ColorF(0.1f, 0.1f, 0.12f));

        D2D1_RECT_F dest = { 0, 0, (float)width, (float)height };
        pRT->DrawBitmap(pBitmap, &dest);

        hr = pRT->EndDraw();
        pBitmap->Release();

        if (hr == D2DERR_RECREATE_TARGET) {
            valid = false;
            pRT->Release(); pRT = nullptr;
        }
    }
};