#include "render.h"
#include <fstream>
#include <sstream>      //std::istringstream
#include <iostream>
#include <vector>
#include <algorithm>

//loadOBJ - 解析OBJ文件并生成三角形数组
int loadOBJ(const char* path, GPUTriangle*& triangles) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[OBJ] 无法打开文件: " << path << std::endl;
        triangles = nullptr;
        return 0;
    }

    std::vector<Vec3> verts;        //顶点集合
    std::vector<GPUTriangle> tris;  //三角型集合（临时）
    std::string line;               //接收每行数据以转换
    int faceIdx = 0;                //面序号
    int lineNum = 0;                //当前行号

    std::cout << "[OBJ]正在读取obj文件...\n";
    //第一步：得到三角形列表
    while (std::getline(file, line)) {
        lineNum++;
        //解析顶点
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            std::istringstream iss(line.substr(2));  //跳过 "v "
            if (iss >> x >> y >> z)
                verts.push_back({ x, y, z });
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            int vi[64];     //顶点索引数组（最多支持64边形）
            int cnt = 0;    //当前面的顶点计数
            std::istringstream iss(line.substr(2));  //跳过 "f "
            std::string token;
            while (iss >> token && cnt < 64) {
                //只读顶点，不读纹理和法线
                int v;
                std::istringstream tss(token);
                tss >> v;
                //obj索引转c++索引
                vi[cnt++] = (v > 0) ? v - 1 : (int)verts.size() + v;
            }
            //如果顶点数超过64个，发出警告（实际会被截断,而且一般不会出现）
            if (cnt >= 64) {
                std::cerr << "[OBJ] 警告: obj文件第 " << lineNum
                    << " 行顶点过多" << std::endl;
            }
            //扇分割：以第一个顶点为公共顶点，依次连接相邻顶点
            for (int i = 1; i + 1 < cnt; ++i) {
                //检查三个顶点索引是否有效（防止越界）
                if (vi[0] < 0 || vi[0] >= (int)verts.size()) continue;
                if (vi[i] < 0 || vi[i] >= (int)verts.size()) continue;
                if (vi[i + 1] < 0 || vi[i + 1] >= (int)verts.size()) continue;

                GPUTriangle tri;
                tri.v[0] = verts[vi[0]];            //扇的公共顶点
                tri.v[1] = verts[vi[i]];
                tri.v[2] = verts[vi[i + 1]];
                Vec3 ab = tri.v[1] - tri.v[0];      //边向量AB
                Vec3 ac = tri.v[2] - tri.v[0];
                tri.normal = ab.cross(ac).norm();   //叉积并归一化得面法线
                //分配颜色
                const unsigned char c[3] = { 190,190,190 };
                tri.color[0] = c[0];   //R分量
                tri.color[1] = c[1];   //G分量
                tri.color[2] = c[2];   //B分量
                tris.push_back(tri);   //将三角形添加到临时数组
                faceIdx++;             //面计数器递增
            }
        }
    }
    file.close();

    //第二步：居中归一化缩放
    if (!tris.empty()) {
        //一：算包围盒（AABB轴对齐）
        Vec3 lo = { FLT_MAX,  FLT_MAX,  FLT_MAX };   //最小点（初始化为最大值）
        Vec3 hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };   //最大点（初始化为最小值）
        for (auto& tri : tris)
            for (int k = 0; k < 3; k++) {
                lo.x = std::min(lo.x, tri.v[k].x);
                lo.y = std::min(lo.y, tri.v[k].y);
                lo.z = std::min(lo.z, tri.v[k].z);
                hi.x = std::max(hi.x, tri.v[k].x);
                hi.y = std::max(hi.y, tri.v[k].y);
                hi.z = std::max(hi.z, tri.v[k].z);
            }
        //二：算中心点
        Vec3 center = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
        //三：选最长的边确定缩放程度，若全为0除法会出错，换成极小数代替
        float maxExt = std::max({ hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-5f });
        //四：缩放因子（使包围盒最长长度度为2，即半径约为1）
        float scale = 2.f / maxExt;
        //五：平移+缩放
#pragma omp parallel for
        for (int i = 0; i < (int)tris.size(); i++) {
            auto& tri = tris[i];
            for (int k = 0; k < 3; k++) {
                tri.v[k].x = (tri.v[k].x - center.x) * scale;
                tri.v[k].y = (tri.v[k].y - center.y) * scale;
                tri.v[k].z = (tri.v[k].z - center.z) * scale;
            }
        }
        //六：重新计算所有面法线保证变换后安全！！！
#pragma omp parallel for
        for (int i = 0; i < (int)tris.size(); i++) {
                auto& tri = tris[i];
            Vec3 ab = tri.v[1] - tri.v[0];
            Vec3 ac = tri.v[2] - tri.v[0];
            tri.normal = ab.cross(ac).norm();
        }
    }

    //第三步：分配输出数组，返回最终三角形数量
    int n = (int)tris.size();
    if (n == 0) {
        fprintf(stderr, "[OBJ] 未发现三角形于: %s\n", path);
        triangles = nullptr;
        return 0;
    }
    //分配内存
    try {
        triangles = new GPUTriangle[n];
    }
    catch (...) {
        fprintf(stderr, "[OBJ] 内存分配失败，共 %d 个三角形\n", n);
        triangles = nullptr;
        return 0;
    }
    //将临时vector中的三角形数据复制到输出数组
    for (int i = 0; i < n; i++) triangles[i] = tris[i];
    //加载成功
    fprintf(stdout, "[OBJ] 加载 %d 顶点, %d 三角形: %s\n",
        (int)verts.size(), n, path);

    return n;
}
