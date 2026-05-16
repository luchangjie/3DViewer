#pragma once
//===================================================================
// app_core.h - 跨文件共享的声明、常量与全局状态
//===================================================================

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include "render.h"
#include "Direct2DDisplay.h"

//-------------------------------------------------------------------
// 前向声明：OBJ / Voxel 加载接口（定义在 obj_loader.cpp / voxel_loader.cpp）
//-------------------------------------------------------------------
int loadOBJ(const char* path, GPUTriangle*& triangles);
int loadVoxelObj(const char* path, GPUTriangle*& triangles);

//-------------------------------------------------------------------
// CUDA DLL 函数指针类型定义
//-------------------------------------------------------------------
typedef int (*cudaInit_t)(int, int, char*, int);
typedef void (*cudaUploadTriangles_t)(const GPUTriangle*, int);
typedef void (*cudaRender_t)(const RenderParams&, unsigned int*);
typedef void (*cudaShutdown_t)(void);
typedef void (*cudaUploadBVH_t)(const BVHNode*, int);

//-------------------------------------------------------------------
// 全局常量
//-------------------------------------------------------------------
extern const int	WIN_W;			// 窗口宽
extern const int	WIN_H;			// 窗口高
extern const float	FOV_Y;			// 垂直视场角（弧度）
extern const float	Z_NEAR;			// 近裁剪面
extern const float  Z_FAR;			// 远裁剪面

//-------------------------------------------------------------------
// CUDA DLL 导出函数指针（全局）
//-------------------------------------------------------------------
extern cudaInit_t				p_cudaInit;
extern cudaUploadTriangles_t	p_cudaUploadTriangles;
extern cudaRender_t				p_cudaRender;
extern cudaShutdown_t			p_cudaShutdown;
extern cudaUploadBVH_t          p_cudaUploadBVH;

extern HMODULE g_hKernel;		// DLL 模块句柄

//-------------------------------------------------------------------
// BVH 全局数据
//-------------------------------------------------------------------
extern std::vector<BVHNode> g_hBVH;		// CPU 端 BVH 节点数组
extern int                  g_numBVHNodes;

//-------------------------------------------------------------------
// 帧缓冲与模型数据
//-------------------------------------------------------------------
extern unsigned int* g_hFramebuf;	// 指向 DIB 位图数据的指针
extern GPUTriangle* g_hTris;		// 主机端三角形数组（CPU 端）
extern int              g_numTris;	// 当前三角形数量

//-------------------------------------------------------------------
// 交互状态（旋转、缩放、平移、模式等）
//-------------------------------------------------------------------
extern float     g_rotX;
extern float     g_rotY;
extern float     g_zoom;
extern bool      g_dragging;
extern int       g_lastMX, g_lastMY;
extern bool      g_panning;
extern float     g_panX, g_panY;
extern bool      g_wireframe;
extern bool      g_voxelMode;
extern std::string g_currentModelPath;

//-------------------------------------------------------------------
// Win32 资源
//-------------------------------------------------------------------
extern HWND         g_hwnd;
extern Direct2DDisplay g_display;

//-------------------------------------------------------------------
// 函数声明
//-------------------------------------------------------------------
bool initRenderer();											// 初始化渲染器（加载 DLL + CUDA）
std::string openFileDialog(HWND owner);							// 打开文件对话框
void buildBVH(GPUTriangle* tris, int numTris, std::vector<BVHNode>& outNodes); // 构建 BVH
void loadModel(const std::string& path);						// 加载模型并上传到 GPU
void initDisplay(HWND hwnd, int w, int h);						// 初始化 Direct2D 显示
void renderFrame();												// 生成并渲染一帧
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp); // 窗口消息回调
