#include "render.h"
#include <algorithm>            // 三改：用于 std::nth_element 构建 BVH
#include <vector>               // 三改：BVH 节点容器
#include <cfloat>

//-------------------------------------------------------------------
// 三改：BVH 构建辅助结构：记录每个三角形的 AABB 与中心点
struct BVHBuildTri {
    int  origIdx;            // 原始三角形索引（重排前）
    Vec3 center;             // 包围盒中心
    Vec3 bmin, bmax;         // 轴对齐包围盒
};
// 递归构建 BVH，返回节点在 outNodes 中的索引
// 构建完成后，tris 数组会被原地重排，使得叶子节点的三角形连续存放
static int buildBVHRecursive(std::vector<BVHBuildTri>& buildTris,
    int start, int end,
    std::vector<BVHNode>& outNodes)
{
    // 【关键】先记录当前节点的索引，再 emplace，避免引用失效
    int nodeIdx = (int)outNodes.size();
    outNodes.emplace_back();            // 分配新节点，但不再持有引用
    // 必须通过 outNodes[nodeIdx] 访问，不能用引用缓存

    // 计算当前区间 [start, end) 内所有三角形的联合 AABB
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
    // 【改】通过索引访问，而非引用
    outNodes[nodeIdx].minX = lo.x;
    outNodes[nodeIdx].minY = lo.y;
    outNodes[nodeIdx].minZ = lo.z;
    outNodes[nodeIdx].maxX = hi.x;
    outNodes[nodeIdx].maxY = hi.y;
    outNodes[nodeIdx].maxZ = hi.z;

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

    // 内部节点：按最长轴分割
    Vec3 extent = hi - lo;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

    int mid = start + count / 2;
    std::nth_element(buildTris.begin() + start,
        buildTris.begin() + mid,
        buildTris.begin() + end,
        [axis](const BVHBuildTri& a, const BVHBuildTri& b) {
            float av = (axis == 0) ? a.center.x : (axis == 1) ? a.center.y : a.center.z;
            float bv = (axis == 0) ? b.center.x : (axis == 1) ? b.center.y : b.center.z;
            return av < bv;
        });

    // 【关键】递归前先记录，递归后通过索引赋值
    // 递归可能再次触发 emplace_back，但 nodeIdx 指向的旧位置仍然有效
    int leftIdx = buildBVHRecursive(buildTris, start, mid, outNodes);
    int rightIdx = buildBVHRecursive(buildTris, mid, end, outNodes);

    outNodes[nodeIdx].left = leftIdx;
    outNodes[nodeIdx].right = rightIdx;
    outNodes[nodeIdx].triStart = -1;
    outNodes[nodeIdx].triCount = 0;
    return nodeIdx;
}
// 对外接口：为三角形数组构建 BVH，并原地重排三角形顺序
// 重排后，BVH 叶子节点的 triStart 直接对应 tris 数组下标，无需二次映射
void buildBVH(GPUTriangle* tris, int numTris, std::vector<BVHNode>& outNodes)
{
    outNodes.clear();
    if (numTris <= 0) return;

    // 1. 先为每个三角形计算 AABB 与中心
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

    // 2. 递归构建 BVH（节点写入 outNodes，同时确定重排顺序）
    buildBVHRecursive(buildTris, 0, numTris, outNodes);

    // 3. 按 buildTris 的顺序原地重排原始三角形数组
    //    使得 BVH 叶子中记录的 triStart/triCount 直接指向连续内存
    std::vector<GPUTriangle> temp(numTris);
    for (int i = 0; i < numTris; ++i) {
        temp[i] = tris[buildTris[i].origIdx];
    }
    for (int i = 0; i < numTris; ++i) {
        tris[i] = temp[i];
    }
}