//===================================================================
//obj_loader.cpp - obj模型加载器
//===================================================================
#include "render.h"
#include <cstdio>//文件操作
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

//-------------------------------------------------------------------
//调色板设置（暂时全灰）
static const unsigned char PALETTE[][3] = {
    {180, 180, 180},   // 浅灰色 R=180, G=180, B=180
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
    {180, 180, 180},   // 浅灰色
};
// 计算调色板中的颜色数量(64位转32位)
static const int PALETTE_SIZE = (int)(sizeof(PALETTE) / sizeof(PALETTE[0]));

//-------------------------------------------------------------------
// loadOBJ - 解析OBJ文件并生成三角形数组
// 参数：
//   path - OBJ文件路径
//   triangles - 输出指针，接收动态分配的三角形数组（调用者负责delete[]）
// 返回值：
//   成功返回三角形数量，失败返回0
int loadOBJ(const char* path, GPUTriangle*& triangles) {
    FILE* fp = fopen(path, "r");//只读模式打开路径指向的文件
    if (!fp) {
        fprintf(stderr, "[OBJ] Cannot open file : % s\n", path);
        triangles = nullptr;
        return 0;
    }

    std::vector<Vec3> verts;        //顶点集合
    std::vector<GPUTriangle> tris;  //三角型集合（临时）
    char line[4096];                //接收每行数据以转换,每行最多4096字符
    int faceIdx = 0;                //面序号
    int lineNum = 0;                //当前行号

    //第一步：得到三角形列表
    while (fgets(line,sizeof(line), fp)) {
        lineNum++;
        //解析顶点
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3)
                verts.push_back({ x,y,z });
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            int vi[64];     // 顶点索引数组（最多支持64边形）
            int cnt = 0;    // 当前面的顶点计数
            char* tok = strtok(line + 2, " \t\n\r");//指针指向被切出的第一个词
            while (tok && cnt < 64) {
                int v;
                sscanf(tok, "%d", &v);//从tok中提取一个整数赋给v
                // OBJ索引从1开始，转换为0基索引负数索引表示从末尾倒数的相对索引
                vi[cnt++] = (v > 0) ? v - 1 : (int)verts.size() + v;
                tok = strtok(nullptr, " \t\n\r");  // 继续切、指
            }
            // 如果顶点数超过64个，发出警告（实际会被截断,而且一般不会出现）
            if (cnt >= 64) {
                fprintf(stderr, "[OBJ] Warning: face at line %d has too many vertices, truncated\n", lineNum);
            }
            // 扇分割：将多边形f 0 1 2 3分解为三角形(0,1,2)和(0,2,3)
            // 以第一个顶点为公共顶点，依次连接相邻顶点
            for (int i = 1; i + 1 < cnt; ++i) {
                // 检查三个顶点索引是否有效（防止越界）
                if (vi[0] < 0 || vi[0] >= (int)verts.size()) continue;
                if (vi[i] < 0 || vi[i] >= (int)verts.size()) continue;
                if (vi[i + 1] < 0 || vi[i + 1] >= (int)verts.size()) continue;

                GPUTriangle tri;
                tri.v[0] = verts[vi[0]];     // 扇的公共顶点
                tri.v[1] = verts[vi[i]];     // 第i个顶点
                tri.v[2] = verts[vi[i + 1]];   // 第i+1个顶点
                Vec3 ab = tri.v[1] - tri.v[0];   // 边向量AB
                Vec3 ac = tri.v[2] - tri.v[0];   // 边向量AC
                tri.normal = ab.cross(ac).norm(); // 叉积并归一化得面法线
                // 分配颜色（从调色板循环选取）
                const unsigned char* c = PALETTE[faceIdx % PALETTE_SIZE];
                tri.color[0] = c[0];   // R分量
                tri.color[1] = c[1];   // G分量
                tri.color[2] = c[2];   // B分量
                tris.push_back(tri);    // 将三角形添加到临时数组
                faceIdx++;              // 面计数器递增
            }
        }
    }
    fclose(fp);

    ////第二步：居中归一化缩放
    if (!tris.empty()) {
        // 一：算包围盒（AABB轴对齐）
        Vec3 lo = { FLT_MAX,  FLT_MAX,  FLT_MAX };   // 最小点（初始化为最大值）
        Vec3 hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };   // 最大点（初始化为最小值）
        for (auto& tri : tris)
            for (int k = 0; k < 3; k++) {
                lo.x = std::min(lo.x, tri.v[k].x);
                lo.y = std::min(lo.y, tri.v[k].y);
                lo.z = std::min(lo.z, tri.v[k].z);
                hi.x = std::max(hi.x, tri.v[k].x);
                hi.y = std::max(hi.y, tri.v[k].y);
                hi.z = std::max(hi.z, tri.v[k].z);
            }//移动最大点和最小点使盒子最匹配
        // 二：算中心点
        Vec3 center = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
        // 三：选最长的边确定缩放程度，若全为0除法会出错，换成极小数代替
        float maxExt = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-5f });
        // 四：缩放因子（使包围盒最长长度度为2，即半径约为1）
        float scale = 2.f / maxExt;
        // 五：平移+缩放
        for (auto& tri : tris)
            for (int k = 0; k < 3; k++) {
                tri.v[k].x = (tri.v[k].x - center.x) * scale;  
                tri.v[k].y = (tri.v[k].y - center.y) * scale;  
                tri.v[k].z = (tri.v[k].z - center.z) * scale;  
            }
        // 六：重新计算所有面法线保证变换后安全！！！
        for (auto& tri : tris) {
            Vec3 ab = tri.v[1] - tri.v[0];   
            Vec3 ac = tri.v[2] - tri.v[0];   
            tri.normal = ab.cross(ac).norm();
        }
    }

    // 第三步：分配输出数组，返回最终三角形数量
    int n = (int)tris.size();
    if (n == 0) {
        fprintf(stderr, "[OBJ] No valid triangles found in: %s\n", path);
        triangles = nullptr;
        return 0;
    }
    //分配内存
    try {
        triangles = new GPUTriangle[n];  // 动态分配三角形数组
    }
    catch (...) {
        // 内存分配失败：输出错误信息
        fprintf(stderr, "[OBJ] Memory allocation failed for %d triangles\n", n);
        triangles = nullptr;
        return 0;
    }
    // 将临时vector中的三角形数据复制到输出数组
    for (int i = 0; i < n; i++) triangles[i] = tris[i];
    // 输出加载成功的统计信息
    fprintf(stdout, "[OBJ] Loaded %d vertices, %d triangles: %s\n",
        (int)verts.size(), n, path);

    return n;
}