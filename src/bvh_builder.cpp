#include "render.h"
#include <algorithm>            // 三改：用于 std::nth_element 构建 BVH
#include <vector>               // 三改：BVH 节点容器
#include <cfloat>

//-------------------------------------------------------------------
// 三改：BVH 构建辅助结构：记录每个三角形的 AABB 与中心点
struct BVHBuildTri {
    int  origIdx;            // 原始三角形索引（重排前）
    Vec3 center;             // 包围盒中心
    Vec3 bmin, bmax;         // 包围盒最小点、最大点
};

// 递归构建 BVH，返回节点在 outNodes 中的索引
static int buildBVHRecursive(
    std::vector<BVHBuildTri>& buildTris,    //要重排的三角形数组
    int start, int end,                     //哪个区间的
    std::vector<BVHNode>& outNodes)         //重排后的BVH节点数组
{
    int nodeIdx = (int)outNodes.size();     // 记录当前节点在数组中的位置
    outNodes.emplace_back();                // 在数组末尾新增一个空节点
    // 必须通过outNodes[nodeIdx]访问，不能用引用缓存，因为缓存可能由于递归扩容而引用失效

    // 计算当前区间 [start, end) 内所有三角形的整体包围盒
    Vec3 lo(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 hi(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = start; i < end; ++i) {
        lo.x = fminf(lo.x, buildTris[i].bmin.x);
        lo.y = fminf(lo.y, buildTris[i].bmin.y);
        lo.z = fminf(lo.z, buildTris[i].bmin.z);
        hi.x = fmaxf(hi.x, buildTris[i].bmax.x);
        hi.y = fmaxf(hi.y, buildTris[i].bmax.y);
        hi.z = fmaxf(hi.z, buildTris[i].bmax.z);
    }
    // 该节点的包围盒
    outNodes[nodeIdx].minX = lo.x;
    outNodes[nodeIdx].minY = lo.y;
    outNodes[nodeIdx].minZ = lo.z;
    outNodes[nodeIdx].maxX = hi.x;
    outNodes[nodeIdx].maxY = hi.y;
    outNodes[nodeIdx].maxZ = hi.z;

    //一个叶子最多包含几个三角形
    int count = end - start;
    const int LEAF_MAX = 8;

    if (count <= LEAF_MAX) {
        // 叶子节点
        outNodes[nodeIdx].left = -1;
        outNodes[nodeIdx].right = -1;
        outNodes[nodeIdx].triStart = start;
        outNodes[nodeIdx].triCount = count;
        return nodeIdx;
    }

    // 内部节点：选最长轴分割
    Vec3 extent = hi - lo;
    int axis = 0;//默认x轴
    if (extent.y > extent.x) axis = 1;
    if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

    int mid = start + count / 2;//中点
    std::nth_element(   //跟中点比较分成两堆
        buildTris.begin() + start,  //起始
        buildTris.begin() + mid,    //中点
        buildTris.begin() + end,    //最后
        [axis](const BVHBuildTri& a, const BVHBuildTri& b) {//比较方法
            float av = (axis == 0) ? a.center.x : (axis == 1) ? a.center.y : a.center.z;// A在最长轴上的坐标
            float bv = (axis == 0) ? b.center.x : (axis == 1) ? b.center.y : b.center.z;// B在最长轴上的坐标
            return av < bv;                                                             // 按坐标从小到大排序
        }
    );

    // 递归，记录
    int leftIdx = buildBVHRecursive(buildTris, start, mid, outNodes);
    int rightIdx = buildBVHRecursive(buildTris, mid, end, outNodes);

    outNodes[nodeIdx].left = leftIdx;
    outNodes[nodeIdx].right = rightIdx;
    outNodes[nodeIdx].triStart = -1;
    outNodes[nodeIdx].triCount = 0;
    return nodeIdx;
}

// 为三角形数组构建BVH，并原地重排三角形顺序
void buildBVH(GPUTriangle* tris, int numTris, std::vector<BVHNode>& outNodes)
{
    outNodes.clear();
    if (numTris <= 0) return;

    // 1. 先为每个三角形计算AABB与中心
    std::vector<BVHBuildTri> buildTris(numTris);
    for (int i = 0; i < numTris; ++i) {
        const GPUTriangle& tri = tris[i];
        Vec3 lo = { fminf(tri.v[0].x, fminf(tri.v[1].x, tri.v[2].x)),
                    fminf(tri.v[0].y, fminf(tri.v[1].y, tri.v[2].y)),
                    fminf(tri.v[0].z, fminf(tri.v[1].z, tri.v[2].z)) };
        Vec3 hi = { fmaxf(tri.v[0].x, fmaxf(tri.v[1].x, tri.v[2].x)),
                    fmaxf(tri.v[0].y, fmaxf(tri.v[1].y, tri.v[2].y)),
                    fmaxf(tri.v[0].z, fmaxf(tri.v[1].z, tri.v[2].z)) };
        buildTris[i].origIdx = i;
        buildTris[i].bmin = lo;
        buildTris[i].bmax = hi;
        buildTris[i].center = (lo + hi) * 0.5f;
    }

    // 2. 递归构建BVH（节点写入outNodes，同时确定重排顺序）
    buildBVHRecursive(buildTris, 0, numTris, outNodes);

    // 3. 按buildTris的顺序重排原始三角形数组tris
    std::vector<GPUTriangle> temp(numTris);
    for (int i = 0; i < numTris; ++i) {
        temp[i] = tris[buildTris[i].origIdx];
    }
    for (int i = 0; i < numTris; ++i) {
        tris[i] = temp[i];
    }
}