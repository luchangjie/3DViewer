# 3DViewer —— CUDA GPU 软件光栅化 3D 模型查看器

一个用 CUDA 在 GPU 上做**逐像素软件光栅化**的 Windows 桌面查看器。支持 OBJ 模型导入、线框 / 实体 / 体素三种显示模式，以及体素化导出。

## 特性

- 纯 CUDA 实现的渲染内核：近裁剪面裁剪、16×16 分块光栅化、Blinn-Phong 光照
- BVH 加速结构（模型空间 AABB + 屏幕空间包围盒剔除）
- 体素化模式（三角–体素 SAT 相交检测，乐高 / MC 风格）
- Direct2D 显示，鼠标 + 键盘交互

## 系统要求

- Windows 10 / 11（64 位）
- NVIDIA 显卡（需安装 NVIDIA 驱动；**编译**还需 CUDA Toolkit）
- Visual Studio 2022（含“使用 C++ 的桌面开发”与 CUDA 支持）

## 构建

- **方式一（推荐）**：双击根目录 `build.bat`，它会自动定位 VS 环境并用 MSBuild 编译 `3DViewer.sln`（`Release | x64`），产物在 `x64/Release/`（`3DViewer.exe` + `render_kernel.dll`）。
- **方式二**：用 Visual Studio 打开 `3DViewer.sln`，直接“生成解决方案”，配置选 `Release | x64`。

> 渲染内核 `render.cu` 由 NVCC 编译为 `render_kernel.dll`，主程序运行时动态加载它，**所以 `3DViewer.exe` 和 `render_kernel.dll` 必须放在同一目录**。

## 使用

| 操作 | 功能 |
| --- | --- |
| 鼠标左键拖拽 | 旋转模型 |
| 鼠标右键拖拽 | 平移视角 |
| 鼠标滚轮 | 缩放（推近 / 拉远） |
| `O` | 打开 OBJ 模型文件 |
| `W` | 切换 线框 / 实体 模式 |
| `V` | 切换 体素模式（需先打开 OBJ） |
| `G` | 导出体素化模型为 OBJ（仅体素模式有效） |
| `↑` / `↓` | 提高 / 降低 体素分辨率（实体模式下调好再切） |
| `R` | 重置视角 |
| `ESC` | 退出 |

窗口标题会实时显示：三角形总数、FPS、当前模式、体素分辨率。

## 项目结构

```
3DViewer/
├─ src/           所有源码（.cpp / .h / .cu）
│   main.cpp          Win32 窗口、消息循环、交互
│   render.cu         CUDA 渲染内核（编译为 DLL）
│   render.h          公共数学(Vec3/Mat4)、结构体、DLL 接口
│   obj_loader.cpp    读取 OBJ
│   voxel_loader.cpp  体素化 + 导出体素 OBJ
│   bvh_builder.cpp   CPU 端 BVH 构建
│   Direct2DDisplay.h Direct2D 显示
├─ render_kernel/  CUDA 工程（render_kernel.vcxproj）
├─ 3DViewer/       主程序工程（3DViewer.vcxproj）
├─ models/         示例模型（*.obj，被 .gitignore 的 *.obj 规则忽略，按需自行放入）
├─ docs/           文档（架构说明.md / 使用说明.md）
├─ README.md       本文件
├─ build.bat       一键构建脚本
└─ 3DViewer.sln    解决方案
```

## 架构简述

详见 [docs/架构说明.md](docs/架构说明.md)。

## 已知问题 / 待改进

- **高体素分辨率（Res > 500）下可能“已构建却不显示”**：分块构建对每个 16×16 像素块能装下的三角形数量有上限（见 `src/render.h` 的 `MAX_TRIS_PER_TILE`），放大模型后会逐渐显示。建议调小上限或改为动态分配。
- ~~漫反射光照坐标系不一致~~：**已修复**。`computeTriCacheKernel` 原先用模型空间法线点乘视图空间光向，旋转模型时漫反射不变；现已改用 `modelView` 把法线转到视图空间（见 `src/render.cu`）。
- 导入的 OBJ 暂不支持材质 / 贴图，模型统一显示为灰色。

## 许可

见仓库 LICENSE（如有）。
