//===================================================================
//voxel_loader.cpp - 表面体素化加载器（C++风格 + 空间哈希加速）
//===================================================================
#include "render.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// 小立方体的8个顶点的3个坐标（中心在原点）
static const float cubeVerts[8][3] = {
    {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
    {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
};

static const int cubeFaces[12][3] = {
    {0, 2, 1}, {0, 3, 2}, // 前面
    {4, 5, 6}, {4, 6, 7}, // 后面
    {0, 1, 5}, {0, 5, 4}, // 底面
    {2, 3, 7}, {2, 7, 6}, // 顶面
    {1, 2, 6}, {1, 6, 5}, // 右面
    {3, 0, 4}, {3, 4, 7}  // 左面
};

// 三角形-AABB 碰撞检测
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
        std::cerr << "[VOXEL] Cannot open file: " << path << std::endl;
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
            int vi[64], cnt = 0;
            std::istringstream iss(line.substr(2));
            std::string token;
            while (iss >> token && cnt < 64) {
                int v;
                std::istringstream tss(token);
                tss >> v;
                if (v > 0) vi[cnt++] = v - 1;
            }
            for (int i = 1; i + 1 < cnt; ++i) {
                if (vi[0] < 0 || vi[0] >= (int)verts.size()) continue;
                if (vi[i] < 0 || vi[i] >= (int)verts.size()) continue;
                if (vi[i + 1] < 0 || vi[i + 1] >= (int)verts.size()) continue;

                GPUTriangle tri;
                tri.v[0] = verts[vi[0]];
                tri.v[1] = verts[vi[i]];
                tri.v[2] = verts[vi[i + 1]];
                Vec3 ab = tri.v[1] - tri.v[0];
                Vec3 ac = tri.v[2] - tri.v[0];
                tri.normal = ab.cross(ac).norm();
                tri.color[0] = 180; tri.color[1] = 180; tri.color[2] = 180;
                tris.push_back(tri);
            }
        }
    }
    file.close();

    if (tris.empty()) {
        std::cerr << "[VOXEL] No triangles in: " << path << std::endl;
        triangles = nullptr;
        return 0;
    }

    // 第二步：计算包围盒
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
    const int resolution = 160;
    float voxelSize = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z }) / resolution;

    // 第四步：构建空间哈希网格
    const int GRID_RES = 40;
    float gridStep = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z }) / GRID_RES;

    std::vector<std::vector<int>> grid(GRID_RES * GRID_RES * GRID_RES);

    for (int t = 0; t < (int)tris.size(); ++t) {
        const GPUTriangle& tri = tris[t];
        float tMinX = fminf(tri.v[0].x, fminf(tri.v[1].x, tri.v[2].x));
        float tMaxX = fmaxf(tri.v[0].x, fmaxf(tri.v[1].x, tri.v[2].x));
        float tMinY = fminf(tri.v[0].y, fminf(tri.v[1].y, tri.v[2].y));
        float tMaxY = fmaxf(tri.v[0].y, fmaxf(tri.v[1].y, tri.v[2].y));
        float tMinZ = fminf(tri.v[0].z, fminf(tri.v[1].z, tri.v[2].z));
        float tMaxZ = fmaxf(tri.v[0].z, fmaxf(tri.v[1].z, tri.v[2].z));

        int gxMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinX - lo.x) / gridStep)));
        int gxMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxX - lo.x) / gridStep)));
        int gyMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinY - lo.y) / gridStep)));
        int gyMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxY - lo.y) / gridStep)));
        int gzMin = std::max(0, std::min(GRID_RES - 1, (int)((tMinZ - lo.z) / gridStep)));
        int gzMax = std::max(0, std::min(GRID_RES - 1, (int)((tMaxZ - lo.z) / gridStep)));

        for (int gz = gzMin; gz <= gzMax; ++gz)
            for (int gy = gyMin; gy <= gyMax; ++gy)
                for (int gx = gxMin; gx <= gxMax; ++gx)
                    grid[(gz * GRID_RES + gy) * GRID_RES + gx].push_back(t);
    }

    for (auto& cell : grid) {
        std::sort(cell.begin(), cell.end());
        cell.erase(std::unique(cell.begin(), cell.end()), cell.end());
    }

    std::cout << "[VOXEL] Grid built: " << GRID_RES * GRID_RES * GRID_RES
        << " cells, avg " << (float)tris.size() / (GRID_RES * GRID_RES * GRID_RES)
        << " tris/cell" << std::endl;

    // 第五步：遍历体素
    std::vector<GPUTriangle> voxelTris;

    for (int ix = 0; ix < resolution; ix++) {
        for (int iy = 0; iy < resolution; iy++) {
            for (int iz = 0; iz < resolution; iz++) {
                float minX = lo.x + ix * voxelSize;
                float minY = lo.y + iy * voxelSize;
                float minZ = lo.z + iz * voxelSize;
                float maxX = minX + voxelSize;
                float maxY = minY + voxelSize;
                float maxZ = minZ + voxelSize;

                int gx = std::max(0, std::min(GRID_RES - 1, (int)((minX - lo.x) / gridStep)));
                int gy = std::max(0, std::min(GRID_RES - 1, (int)((minY - lo.y) / gridStep)));
                int gz = std::max(0, std::min(GRID_RES - 1, (int)((minZ - lo.z) / gridStep)));
                const auto& cell = grid[(gz * GRID_RES + gy) * GRID_RES + gx];

                bool hit = false;
                for (int t : cell) {
                    if (triangleAABBIntersect(tris[t], minX, minY, minZ, maxX, maxY, maxZ)) {
                        hit = true;
                        break;
                    }
                }
                if (!hit) continue;

                float cx = (minX + maxX) * 0.5f;
                float cy = (minY + maxY) * 0.5f;
                float cz = (minZ + maxZ) * 0.5f;

                for (int f = 0; f < 12; f++) {
                    GPUTriangle vTri;
                    for (int k = 0; k < 3; k++) {
                        int vidx = cubeFaces[f][k];
                        vTri.v[k].x = cx + cubeVerts[vidx][0] * voxelSize;
                        vTri.v[k].y = cy + cubeVerts[vidx][1] * voxelSize;
                        vTri.v[k].z = cz + cubeVerts[vidx][2] * voxelSize;
                    }
                    Vec3 ab = vTri.v[1] - vTri.v[0];
                    Vec3 ac = vTri.v[2] - vTri.v[0];
                    vTri.normal = ab.cross(ac).norm();
                    vTri.color[0] = 180; vTri.color[1] = 180; vTri.color[2] = 180;
                    voxelTris.push_back(vTri);
                }
            }
        }
    }

    if (voxelTris.empty()) {
        std::cerr << "[VOXEL] No surface voxels generated" << std::endl;
        triangles = nullptr;
        return 0;
    }

    // 第六步：居中归一化
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
    Vec3 center = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
    float maxExt = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-5f });
    float scale = 2.0f / maxExt;

    for (auto& tri : voxelTris) {
        for (int k = 0; k < 3; k++) {
            tri.v[k].x = (tri.v[k].x - center.x) * scale;
            tri.v[k].y = (tri.v[k].y - center.y) * scale;
            tri.v[k].z = (tri.v[k].z - center.z) * scale;
        }
        Vec3 ab = tri.v[1] - tri.v[0];
        Vec3 ac = tri.v[2] - tri.v[0];
        tri.normal = ab.cross(ac).norm();
    }

    // 第七步：输出数组
    int n = (int)voxelTris.size();
    try {
        triangles = new GPUTriangle[n];
    }
    catch (...) {
        std::cerr << "[VOXEL] Memory allocation failed" << std::endl;
        triangles = nullptr;
        return 0;
    }
    for (int i = 0; i < n; i++) triangles[i] = voxelTris[i];

    std::cout << "[VOXEL] Surface voxels: " << n / 12
        << " cubes, " << n << " triangles from: " << path << std::endl;
    return n;
}