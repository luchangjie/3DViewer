//===================================================================
//voxel_loader.cpp - 表面体素化加载器
//===================================================================
#include "render.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// 小立方体的8个顶点的3个坐标（中心在原点）
static const float cubeVerts[8][3] = {
    {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f}, { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
    {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f}, { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}
};
// 小立方体的12个三角形各自的3顶点
static const int cubeFaces[12][3] = {
    {0, 2, 1}, {0, 3, 2}, // 前面
    {4, 5, 6}, {4, 6, 7}, // 后面
    {0, 1, 5}, {0, 5, 4}, // 底面
    {2, 3, 7}, {2, 7, 6}, // 顶面
    {1, 2, 6}, {1, 6, 5}, // 右面
    {3, 0, 4}, {3, 4, 7}  // 左面
};
// 三角形-AABB 碰撞检测（用包围盒快速判断体素是否和三角形相交）
static bool triangleAABBIntersect(const GPUTriangle& tri,
    //体素俩顶点
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ) {
    //三角形顶点
    const Vec3& v0 = tri.v[0];
    const Vec3& v1 = tri.v[1];
    const Vec3& v2 = tri.v[2];
    //三角形包围盒俩顶点
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
    // 第一步：解析 OBJ 文件，得到三角形列表
    std::vector<Vec3> verts;
    std::vector<GPUTriangle> tris;

    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[VOXEL] Cannot open file: %s\n", path);
        triangles = nullptr;
        return 0;
    }

    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3)
                verts.push_back({ x, y, z });
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            int vi[64], cnt = 0;
            char* tok = strtok(line + 2, " \t\n\r");
            while (tok && cnt < 64) {
                int v;
                sscanf(tok, "%d", &v);
                if (v > 0) vi[cnt++] = v - 1;
                tok = strtok(nullptr, " \t\n\r");
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
                tri.color[0] = 180; tri.color[1] = 180; tri.color[2] = 180;//体素默认灰色，故相比于obj_loader少了调色板
                tris.push_back(tri);
            }
        }
    }
    fclose(fp);

    if (tris.empty()) {
        fprintf(stderr, "[VOXEL] No triangles in: %s\n", path);
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

    // 第三步：体素化参数
    const int resolution =5;                  //体素密度（类似分辨率）
    float stepX = (hi.x - lo.x) / resolution;
    float stepY = (hi.y - lo.y) / resolution;
    float stepZ = (hi.z - lo.z) / resolution;   //体素步长
    float voxelSize = std::max({ stepX, stepY, stepZ });    //选一个当边长使其成正方体

    // 第四步：遍历体素，检测和三角形相交，生成表面立方体
    std::vector<GPUTriangle> voxelTris;         //要生成的体素三角形数组
    for (int ix = 0; ix < resolution; ix++) {
        for (int iy = 0; iy < resolution; iy++) {
            for (int iz = 0; iz < resolution; iz++) {
                float minX = lo.x + ix * voxelSize;
                float minY = lo.y + iy * voxelSize;
                float minZ = lo.z + iz * voxelSize;
                float maxX = minX + voxelSize;
                float maxY = minY + voxelSize;
                float maxZ = minZ + voxelSize;
                // 检查体素是否和任意三角形相交
                bool hit = false;
                for (const auto& tri : tris) {
                    if (triangleAABBIntersect(tri, minX, minY, minZ, maxX, maxY, maxZ)) {
                        hit = true;
                        break;
                    }
                }
                if (!hit) continue;
                // 若hit不为true，在体素中心画一个小立方体（12 个三角形）
                //中心坐标
                float cx = (minX + maxX) * 0.5f;
                float cy = (minY + maxY) * 0.5f;
                float cz = (minZ + maxZ) * 0.5f;

                for (int f = 0; f < 12; f++) {          //12个三角形
                    GPUTriangle vTri;           //对当前三角形
                    for (int k = 0; k < 3; k++) {       //3个顶点
                        int vidx = cubeFaces[f][k];     //每个顶点的世界坐标
                        vTri.v[k].x = cx + cubeVerts[vidx][0] * voxelSize;
                        vTri.v[k].y = cy + cubeVerts[vidx][1] * voxelSize;
                        vTri.v[k].z = cz + cubeVerts[vidx][2] * voxelSize;
                    }
                    Vec3 ab = vTri.v[1] - vTri.v[0];    //边向量AB
                    Vec3 ac = vTri.v[2] - vTri.v[0];    //边向量AC
                    vTri.normal = ab.cross(ac).norm();  //面法线
                    vTri.color[0] = 180; vTri.color[1] = 180; vTri.color[2] = 180;
                    voxelTris.push_back(vTri);          //存入数组
                }
            }
        }
    }
    if (voxelTris.empty()) {
        fprintf(stderr, "[VOXEL] No surface voxels generated\n");
        triangles = nullptr;
        return 0;
    }
    // 第五步：居中并归一化
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
    // 第六步：分配输出数组
    int n = (int)voxelTris.size();
    try {
        triangles = new GPUTriangle[n];
    }
    catch (...) {
        fprintf(stderr, "[VOXEL] Memory allocation failed\n");
        triangles = nullptr;
        return 0;
    }
    for (int i = 0; i < n; i++) triangles[i] = voxelTris[i];

    printf("[VOXEL] Surface voxels: %d cubes, %d triangles from: %s\n",
        n / 12, n, path);
    return n;
}


