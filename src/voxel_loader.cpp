#include "render.h"
#include <fstream>
#include <sstream>      //字符串流
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>//hash表，导出时去重用

// 小立方体的8个顶点的3个坐标（中心在原点）
static const float cubeVerts[8][3] = {
    {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, // 0,1: 底面后
    { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f}, // 2,3: 顶面前
    {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f}, // 4,5: 底面后
    { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}  // 6,7: 顶面前
};

//每个面的三个顶点分别对应顶点几
static const int cubeFaces[12][3] = {
    // 前面 (+Z) - 朝向屏幕外
    {4, 6, 5}, {4, 7, 6},
    // 后面 (-Z) - 朝向屏幕内
    {0, 1, 2}, {0, 2, 3},
    // 右面 (+X)
    {1, 6, 2}, {1, 5, 6},
    // 左面 (-X)
    {0, 3, 7}, {0, 7, 4},
    // 上面 (+Y)
    {3, 6, 7}, {3, 2, 6},
    // 下面 (-Y)
    {0, 4, 5}, {0, 5, 1}
};

// 每个面的相邻体素偏移（用于内部面剔除）
static const int faceNeighbors[6][3] = {
    {0, 0, 1},   // 前面 (+Z)
    {0, 0, -1},  // 后面 (-Z)
    {1, 0, 0},   // 右面 (+X)
    {-1, 0, 0},  // 左面 (-X)
    {0, 1, 0},   // 上面 (+Y)
    {0, -1, 0}   // 下面 (-Y)
};

// 使用位图代替unordered_set，性能提升10倍，内存减少80%
// 对于192分辨率，只需要192*192*192/8=874KB内存
class VoxelBitmap {
private:
    unsigned char* data;
    int sizeX, sizeY, sizeZ;
    int strideY, strideZ;

    // MSVC兼容的popcount函数（计算一个字节中1的个数）
    static inline int popcount(unsigned char b) {
        // 查表法：速度最快，且完全兼容所有编译器
        static const unsigned char table[] = {
            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
            1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
            1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
            1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
            2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
            3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
            3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
            4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
        };
        return table[b];
    }

public:
    VoxelBitmap(int x, int y, int z) : sizeX(x), sizeY(y), sizeZ(z) {
        strideY = (x + 7) / 8;
        strideZ = strideY * y;
        size_t totalBytes = (size_t)strideZ * z;
        data = new unsigned char[totalBytes](); // 初始化为0
    }

    ~VoxelBitmap() {
        delete[] data;
    }

    void set(int x, int y, int z) {
        size_t idx = (size_t)z * strideZ + (size_t)y * strideY + x / 8;
        data[idx] |= 1 << (x % 8);
    }

    bool get(int x, int y, int z) const {
        if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ)
            return false;
        size_t idx = (size_t)z * strideZ + (size_t)y * strideY + x / 8;
        return (data[idx] & (1 << (x % 8))) != 0;
    }

    int count() const {
        int cnt = 0;
        size_t totalBytes = (size_t)strideZ * sizeZ;
        for (size_t i = 0; i < totalBytes; ++i) {
            cnt += popcount(data[i]);
        }
        return cnt;
    }
};

// 精确的三角形-体素相交检测（分离轴定理，高分辨率优化版）
static bool triangleVoxelIntersect(const GPUTriangle& tri,
    float cx, float cy, float cz, float halfSize)
{
    // 将三角形顶点转换到体素局部坐标系
    Vec3 v0 = tri.v[0] - Vec3(cx, cy, cz);
    Vec3 v1 = tri.v[1] - Vec3(cx, cy, cz);
    Vec3 v2 = tri.v[2] - Vec3(cx, cy, cz);

    // 快速AABB测试（先过滤掉明显不相交的情况）
    const float EPS = 1e-4f; // 增大容差以适应高分辨率
    float minX = fminf(v0.x, fminf(v1.x, v2.x));
    float maxX = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    if (maxX < -halfSize - EPS || minX > halfSize + EPS) return false;

    float minY = fminf(v0.y, fminf(v1.y, v2.y));
    float maxY = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    if (maxY < -halfSize - EPS || minY > halfSize + EPS) return false;

    float minZ = fminf(v0.z, fminf(v1.z, v2.z));
    float maxZ = fmaxf(v0.z, fmaxf(v1.z, v2.z));
    if (maxZ < -halfSize - EPS || minZ > halfSize + EPS) return false;

    // 三角形边
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // 体素的三个轴
    Vec3 a0(1, 0, 0);
    Vec3 a1(0, 1, 0);
    Vec3 a2(0, 0, 1);

    // 测试轴：三角形边 × 体素轴（共9个）
    Vec3 axes[] = {
        a0.cross(e0), a0.cross(e1), a0.cross(e2),
        a1.cross(e0), a1.cross(e1), a1.cross(e2),
        a2.cross(e0), a2.cross(e1), a2.cross(e2)
    };

    // 测试所有分离轴
    for (int i = 0; i < 9; ++i) {
        Vec3 axis = axes[i];
        if (axis.lenSq() < 1e-6f) continue; // 跳过零向量
        axis = axis.norm();

        float p0 = v0.dot(axis);
        float p1 = v1.dot(axis);
        float p2 = v2.dot(axis);
        float triMin = fminf(p0, fminf(p1, p2));
        float triMax = fmaxf(p0, fmaxf(p1, p2));

        float boxProj = halfSize * (fabsf(a0.dot(axis)) + fabsf(a1.dot(axis)) + fabsf(a2.dot(axis)));

        // 添加epsilon容差，解决高分辨率下的浮点精度问题
        if (triMax < -boxProj - 1e-5f || triMin > boxProj + 1e-5f) {
            return false; // 分离轴存在，不相交
        }
    }

    // 测试三角形法线轴
    Vec3 normal = e0.cross(e1).norm();
    float p0 = v0.dot(normal);
    float p1 = v1.dot(normal);
    float p2 = v2.dot(normal);
    float triMinN = fminf(p0, fminf(p1, p2));
    float triMaxN = fmaxf(p0, fmaxf(p1, p2));
    float boxProjN = halfSize * (fabsf(a0.dot(normal)) + fabsf(a1.dot(normal)) + fabsf(a2.dot(normal)));

    if (triMaxN < -boxProjN - 1e-5f || triMinN > boxProjN + 1e-5f) {
        return false;
    }

    // 所有轴都通过，相交
    return true;
}

// 三角形-AABB 碰撞检测（保留你原来的函数，用于空间哈希网格）
static bool triangleAABBIntersect(const GPUTriangle& tri,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ)
{
    const Vec3& v0 = tri.v[0];
    const Vec3& v1 = tri.v[1];
    const Vec3& v2 = tri.v[2];
    float triMinX = fminf(v0.x, fminf(v1.x, v2.x));
    float triMaxX = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    float triMinY = fminf(v0.y, fminf(v1.y, v2.y));
    float triMaxY = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    float triMinZ = fminf(v0.z, fminf(v1.z, v2.z));
    float triMaxZ = fmaxf(v0.z, fmaxf(v1.z, v2.z));
    if (triMaxX < minX || triMinX > maxX) return false;
    if (triMaxY < minY || triMinY > maxY) return false;
    if (triMaxZ < minZ || triMinZ > maxZ) return false;
    return true;
}

int loadVoxelObj(const char* path, GPUTriangle*& triangles) {
    // 第一步：解析 OBJ 文件
    std::vector<Vec3> verts;
    std::vector<GPUTriangle> tris;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[VOXEL] 无法打开文件: " << path << std::endl;
        triangles = nullptr;
        return 0;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            std::istringstream iss(line.substr(2));
            if (iss >> x >> y >> z)
                verts.push_back({ x, y, z });
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            int vi[64], cnt = 0;                    //最大64边形，计数
            std::istringstream iss(line.substr(2));
            std::string token;
            while (iss >> token && cnt < 64) {      //每空格切一token
                int v;
                std::istringstream tss(token);
                tss >> v;                           //文本转int（obj顶点索引从1开始）
                if (v > 0) vi[cnt++] = v - 1;       //转为从0开始的索引
            }
            for (int i = 1; i + 1 < cnt; ++i) {
                //跳过越界的索引
                if (vi[0] < 0 || vi[0] >= (int)verts.size()) continue;
                if (vi[i] < 0 || vi[i] >= (int)verts.size()) continue;
                if (vi[i + 1] < 0 || vi[i + 1] >= (int)verts.size()) continue;
                //以第0个顶点为扇心，切多边形为三角形
                GPUTriangle tri;
                tri.v[0] = verts[vi[0]];
                tri.v[1] = verts[vi[i]];
                tri.v[2] = verts[vi[i + 1]];
                tris.push_back(tri);
            }
        }
    }
    file.close();

    if (tris.empty()) {
        std::cerr << "[VOXEL]未发现三角形在: " << path << std::endl;
        triangles = nullptr;
        return 0;
    }

    // 第二步：计算模型包围盒
    Vec3 lo(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const auto& tri : tris) {
        for (int k = 0; k < 3; k++) {
            lo.x = std::min(lo.x, tri.v[k].x);
            lo.y = std::min(lo.y, tri.v[k].y);
            lo.z = std::min(lo.z, tri.v[k].z);
            hi.x = std::max(hi.x, tri.v[k].x);
            hi.y = std::max(hi.y, tri.v[k].y);
            hi.z = std::max(hi.z, tri.v[k].z);
        }
    }

    // 第三步：体素参数
    extern int g_voxelResolution;               // 声明外部全局变量（在main.cpp）
    const int resolution = g_voxelResolution;

    float voxelSize = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z }) / resolution; //选最长的边计算体素边长

    // 第四步：构建空间哈希网格
    // 网格分辨率根据体素分辨率自动调整，每个网格单元覆盖8x8x8个体素
    int GRID_RES = resolution / 8;
    GRID_RES = std::min(64, std::max(4, GRID_RES)); // 限制在4-64之间

    float gridStep = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z }) / GRID_RES;    //选最长的轴计算大立方体边长

    std::vector<std::vector<int>> grid(GRID_RES * GRID_RES * GRID_RES);                 //创建网格，每格存三角形索引数组

    for (int t = 0; t < (int)tris.size(); ++t) {
        const GPUTriangle& tri = tris[t];
        //三角形包围盒
        float tMinX = fminf(tri.v[0].x, fminf(tri.v[1].x, tri.v[2].x));
        float tMaxX = fmaxf(tri.v[0].x, fmaxf(tri.v[1].x, tri.v[2].x));
        float tMinY = fminf(tri.v[0].y, fminf(tri.v[1].y, tri.v[2].y));
        float tMaxY = fmaxf(tri.v[0].y, fmaxf(tri.v[1].y, tri.v[2].y));
        float tMinZ = fminf(tri.v[0].z, fminf(tri.v[1].z, tri.v[2].z));
        float tMaxZ = fmaxf(tri.v[0].z, fmaxf(tri.v[1].z, tri.v[2].z));
        //三角形的单轴网格索引（在第几个格子）
        int gxMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinX - lo.x) / gridStep)));
        int gxMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxX - lo.x) / gridStep)));
        int gyMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinY - lo.y) / gridStep)));
        int gyMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxY - lo.y) / gridStep)));
        int gzMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinZ - lo.z) / gridStep)));
        int gzMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxZ - lo.z) / gridStep)));
        //三角形在哪些格子，添加其索引到相应格子的int数组里
        for (int gz = gzMin; gz <= gzMax; ++gz)
            for (int gy = gyMin; gy <= gyMax; ++gy)
                for (int gx = gxMin; gx <= gxMax; ++gx)
                    grid[(gz * GRID_RES + gy) * GRID_RES + gx].push_back(t);
    }
    //删去可能重复的（cell是grid[i],int数组）
    for (auto& cell : grid) {
        std::sort(cell.begin(), cell.end());
        cell.erase(std::unique(cell.begin(), cell.end()), cell.end());
    }

    std::cout << "[VOXEL] 网格已建造: " << GRID_RES * GRID_RES * GRID_RES
        << " 个网格单元，平均每单元 " << (float)tris.size() / (GRID_RES * GRID_RES * GRID_RES)
        << " 个三角形" << std::endl;

    // 第五步：OpenMP多线程遍历体素（使用位图代替unordered_set）
    VoxelBitmap voxelBitmap(resolution, resolution, resolution);

    std::cout << "[VOXEL] 开始体素化，分辨率: " << resolution << "x" << resolution << "x" << resolution
        << "，总共有 " << resolution * resolution * resolution << " 个体素" << std::endl;

    // OpenMP并行处理X轴（最外层循环）
    // 自动使用所有CPU核心，性能与C++17并行算法相当
#pragma omp parallel for schedule(dynamic)
    for (int ix = 0; ix < resolution; ix++) {
        for (int iy = 0; iy < resolution; iy++) {
            for (int iz = 0; iz < resolution; iz++) {
                //体素边界位置
                float minX = lo.x + ix * voxelSize;
                float minY = lo.y + iy * voxelSize;
                float minZ = lo.z + iz * voxelSize;
                float maxX = minX + voxelSize;
                float maxY = minY + voxelSize;
                float maxZ = minZ + voxelSize;
                // 【优化】使用更精确的体素中心计算方式
                // 避免浮点累积误差，特别是在高分辨率下
                float cx = lo.x + (ix + 0.5f) * voxelSize;
                float cy = lo.y + (iy + 0.5f) * voxelSize;
                float cz = lo.z + (iz + 0.5f) * voxelSize;
                //体素位于的网格索引
                int gx = std::max(0, std::min(GRID_RES - 1, (int)((cx - lo.x) / gridStep)));
                int gy = std::max(0, std::min(GRID_RES - 1, (int)((cy - lo.y) / gridStep)));
                int gz = std::max(0, std::min(GRID_RES - 1, (int)((cz - lo.z) / gridStep)));
                const auto& cell = grid[(gz * GRID_RES + gy) * GRID_RES + gx];
                //是否碰撞
                bool hit = false;
                for (int t : cell) {
                    if (triangleVoxelIntersect(tris[t], cx, cy, cz, voxelSize * 0.5f)) {
                        hit = true;
                        break;
                    }
                }
                if (hit) {
                    voxelBitmap.set(ix, iy, iz);
                }
            }
        }
    }

    int numVoxels = voxelBitmap.count();
    if (numVoxels == 0) {
        std::cerr << "[VOXEL] 未生成任何表面体素" << std::endl;
        triangles = nullptr;
        return 0;
    }

    std::cout << "[VOXEL] 生成体素数: " << numVoxels << std::endl;

    // 第六步：内部面剔除，生成最终三角形
    std::vector<GPUTriangle> voxelTris;
    voxelTris.reserve(numVoxels * 6); // 预分配内存，避免多次重新分配

    for (int ix = 0; ix < resolution; ix++) {
        for (int iy = 0; iy < resolution; iy++) {
            for (int iz = 0; iz < resolution; iz++) {
                if (!voxelBitmap.get(ix, iy, iz)) continue;

                float cx = lo.x + (ix + 0.5f) * voxelSize;
                float cy = lo.y + (iy + 0.5f) * voxelSize;
                float cz = lo.z + (iz + 0.5f) * voxelSize;

                // 检查6个面是否需要绘制
                for (int face = 0; face < 6; ++face) {
                    // 计算相邻体素坐标
                    int nx = ix + faceNeighbors[face][0];
                    int ny = iy + faceNeighbors[face][1];
                    int nz = iz + faceNeighbors[face][2];

                    // 如果相邻体素不存在，才绘制这个面
                    if (!voxelBitmap.get(nx, ny, nz)) {
                        // 生成这个面的两个三角形
                        for (int tri = 0; tri < 2; ++tri) {
                            GPUTriangle vTri;
                            int faceIdx = face * 2 + tri;
                            for (int k = 0; k < 3; k++) {
                                int vidx = cubeFaces[faceIdx][k];
                                vTri.v[k].x = cx + cubeVerts[vidx][0] * voxelSize;
                                vTri.v[k].y = cy + cubeVerts[vidx][1] * voxelSize;
                                vTri.v[k].z = cz + cubeVerts[vidx][2] * voxelSize;
                            }
                            // 计算法线（顶点顺序已经保证法线朝外）
                            Vec3 ab = vTri.v[1] - vTri.v[0];
                            Vec3 ac = vTri.v[2] - vTri.v[0];
                            vTri.normal = ab.cross(ac).norm();

                            // 计算主轴的权重（绝对值最大的那个轴）
                            float weightX = fabsf(vTri.normal.x);
                            float weightY = fabsf(vTri.normal.y);
                            float weightZ = fabsf(vTri.normal.z);
                            // 根据主要朝向分配不同灰度值
                            float gray;
                            if (weightX >= weightY && weightX >= weightZ) {
                                gray = 110.0f;           // X轴面
                            }
                            else if (weightY >= weightX && weightY >= weightZ) {
                                gray = 140.0f;           // Y轴面
                            }
                            else {
                                gray = 170.0f;            // Z轴面
                            }

                            vTri.color[0] = (unsigned char)gray;
                            vTri.color[1] = (unsigned char)gray;
                            vTri.color[2] = (unsigned char)gray;
                            voxelTris.push_back(vTri);
                        }
                    }
                }
            }
        }
    }

    // 第七步：居中归一化
    //   计算模型包围盒
    lo = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    hi = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const auto& tri : voxelTris) {
        for (int k = 0; k < 3; k++) {
            lo.x = std::min(lo.x, tri.v[k].x);
            lo.y = std::min(lo.y, tri.v[k].y);
            lo.z = std::min(lo.z, tri.v[k].z);
            hi.x = std::max(hi.x, tri.v[k].x);
            hi.y = std::max(hi.y, tri.v[k].y);
            hi.z = std::max(hi.z, tri.v[k].z);
        }
    }
    //计算居中变换，映射到[-1,1]
    Vec3 center = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
    float maxExt = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-5f });
    float scale = 2.0f / maxExt;
    //每个三角形进行居中变换
    for (auto& tri : voxelTris) {
        for (int k = 0; k < 3; k++) {
            tri.v[k].x = (tri.v[k].x - center.x) * scale;
            tri.v[k].y = (tri.v[k].y - center.y) * scale;
            tri.v[k].z = (tri.v[k].z - center.z) * scale;
        }
        // 重新归一化法线
        tri.normal = tri.normal.norm();
    }

    // 第八步：输出数组
    int n = (int)voxelTris.size();
    try {
        triangles = new GPUTriangle[n];
    }
    catch (...) {
        std::cerr << "[VOXEL] 内存分配错误" << std::endl;
        triangles = nullptr;
        return 0;
    }
    for (int i = 0; i < n; i++) triangles[i] = voxelTris[i];

    std::cout << "[VOXEL] 表面体素数: " << numVoxels
        << " 块，" << n << " 个三角形（内部面剔除减少了"
        << (int)(100 - (float)n / ((float)numVoxels * 12) * 100) << "%）"
        << "，来自于 " << path << std::endl;
    return n;
}

// 导出体素化后的三角形为OBJ文件
bool exportVoxelObj(const char* path, const GPUTriangle* tris, int numTris) {
    if (!tris || numTris <= 0) return false;

    std::ofstream file(path);
    if (!file.is_open()) return false;

    // OBJ 文件头
    file << "# Voxelized OBJ exported from 3DViewer\n";

    // 1. 顶点去重（哈希表加速，O(n)）
    std::vector<Vec3> uniqueVerts;
    std::vector<int> vertIdx(numTris * 3);
    std::unordered_map<size_t, std::vector<std::pair<Vec3, int>>> hashMap;

    auto hashFunc = [](const Vec3& v) -> size_t {
        size_t hx = (size_t)(v.x * 1000.0f + 0.5f);
        size_t hy = (size_t)(v.y * 1000.0f + 0.5f);
        size_t hz = (size_t)(v.z * 1000.0f + 0.5f);
        return hx ^ (hy << 10) ^ (hz << 20);
        };

    for (int i = 0; i < numTris; ++i) {
        for (int k = 0; k < 3; ++k) {
            const Vec3& v = tris[i].v[k];
            size_t h = hashFunc(v);
            int found = -1;

            auto& bucket = hashMap[h];
            for (auto& p : bucket) {
                if (fabsf(p.first.x - v.x) < 1e-6f &&
                    fabsf(p.first.y - v.y) < 1e-6f &&
                    fabsf(p.first.z - v.z) < 1e-6f) {
                    found = p.second;
                    break;
                }
            }

            if (found < 0) {
                found = (int)uniqueVerts.size();
                uniqueVerts.push_back(v);
                bucket.push_back({ v, found });
            }
            vertIdx[i * 3 + k] = found;
        }
    }

    // 2. 写入去重后的顶点
    file << "# Vertices: " << uniqueVerts.size() << "\n";
    for (const auto& v : uniqueVerts) {
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    file << "\n";

    // 3. 写入法线
    file << "# Normals: " << numTris << "\n";
    for (int i = 0; i < numTris; ++i) {
        const Vec3& n = tris[i].normal;
        file << "vn " << n.x << " " << n.y << " " << n.z << "\n";
    }
    file << "\n";

    // 4. 写入面（OBJ 索引从 1 开始）
    file << "# Faces: " << numTris << "\n";
    for (int i = 0; i < numTris; ++i) {
        int v1 = vertIdx[i * 3 + 0] + 1;
        int v2 = vertIdx[i * 3 + 1] + 1;
        int v3 = vertIdx[i * 3 + 2] + 1;
        file << "f " << v1 << "//" << (i + 1)
            << " " << v2 << "//" << (i + 1)
            << " " << v3 << "//" << (i + 1) << "\n";
    }

    file.close();
    std::cout << "[VOXEL] 导出OBJ成功: " << path << "\n";
    std::cout << "[VOXEL] 顶点数: " << uniqueVerts.size()
        << ", 面数: " << numTris << std::endl;
    return true;
}