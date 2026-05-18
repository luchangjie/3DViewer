//===================================================================
//render.h - 头文件
//===================================================================
#ifndef __CUDACC__
# ifndef __host__
#  define __host__
# endif
# ifndef __device__
#  define __device__
# endif
# ifndef __forceinline__   //如果 __forceinline__ 未定义
#  define __forceinline__ inline  //用标准 inline 替代
# endif
#endif

#include <cmath>
#include <cstring>
#include <cfloat>//浮点极限值头文件，obj_loader.cpp算包围盒用

//二维坐标
struct Vec2 {
    float x, y;
    __host__ __device__ Vec2(float x = 0, float y = 0) :x(x), y(y) {}//构造函数
};

//三维坐标（不是矩阵）
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0) :x(x), y(y), z(z) {}//构造函数
    //向量加法
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return { x + b.x, y + b.y, z + b.z }; }
    //向量减法
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return { x - b.x, y - b.y, z - b.z }; }
    //标量乘法（缩放）
    __host__ __device__ Vec3 operator*(float t)        const { return { x * t,   y * t,   z * t }; }
    //取反（负向量）
    __host__ __device__ Vec3 operator-()               const { return { -x,    -y,    -z }; }
    //点积：比方向（>0同向，=0垂直，<0反向）
    __host__ __device__ float dot(const Vec3& b)  const { return x * b.x + y * b.y + z * b.z; }
    //叉积
    __host__ __device__ Vec3  cross(const Vec3& b) const {
        return { y * b.z - z * b.y,
                 z * b.x - x * b.z,
                 x * b.y - y * b.x };
    }
    //模长
    __host__ __device__ float len()   const { return sqrtf(x * x + y * y + z * z); }
    //模长平方
    __host__ __device__ float lenSq()   const { return (x * x + y * y + z * z); }
    //归一化标准化，零向量返回(0,1,0)，使后续三角形法线方向默认朝上，不影响视觉
    __host__ __device__ Vec3  norm()  const {
        float l = len();
        return (l > 1e-7f) ? Vec3(x / l, y / l, z / l) : Vec3(0, 1, 0);
    }
};

//四维矩阵，进行3D变换
struct Mat4 {
    float m[4][4];//m[col][row]
    __host__ __device__ Mat4() { memset(m, 0, sizeof(m)); }
    //单位矩阵，即无变换
    __host__ __device__ static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f; return r;
    }
    //矩阵乘法
    __host__ __device__ Mat4 operator*(const Mat4& b) const {
        Mat4 r;//对于结果矩阵r
        for (int col = 0; col < 4; col++)
            for (int row = 0; row < 4; row++) {
                float s = 0;
                for (int k = 0; k < 4; k++)
                    s += m[k][row] * b.m[col][k];//左k行*右列k挨个求和
                r.m[col][row] = s;
            }
        return r;
    }
    //点的位置变换（平移旋转缩放）
    __host__ __device__ Vec3 mulPoint(const Vec3& v) const {//三维变四维（x,y,z,w=1）
        float x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
        float y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
        float z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];
        float w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3];
        //透视除法，防0向量
        return (fabsf(w) > 1e-7f) ? Vec3(x / w, y / w, z / w) : Vec3(x, y, z);
    }
    //方向向量变换
    __host__ __device__ Vec3 mulDir(const Vec3& v) const {
        return { m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
                 m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
                 m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z };
    }

    //创建用于变换的矩阵
    //平移矩阵：将物体沿各轴移动
    static Mat4 translate(float tx, float ty, float tz) {
        Mat4 r = identity();
        r.m[3][0] = tx; r.m[3][1] = ty; r.m[3][2] = tz;  //设置平移分量
        return r;
    }
    //缩放矩阵：沿各轴拉伸/压缩
    static Mat4 scale(float sx, float sy, float sz) {
        Mat4 r = identity();
        r.m[0][0] = sx; r.m[1][1] = sy; r.m[2][2] = sz;  //对角线设置缩放因子
        return r;
    }
    //绕Y轴旋转矩阵（弧度制）类比球坐标系的0-2PAI
    static Mat4 rotateY(float a) {
        Mat4 r = identity();
        r.m[0][0] = cosf(a); r.m[2][0] = sinf(a);   //旋转X和Z分量
        r.m[0][2] = -sinf(a); r.m[2][2] = cosf(a);
        return r;
    }
    //绕X轴旋转矩阵（弧度制）类比球坐标系的0-PAI
    static Mat4 rotateX(float a) {
        Mat4 r = identity();
        r.m[1][1] = cosf(a); r.m[2][1] = -sinf(a);   //旋转Y和Z分量
        r.m[1][2] = sinf(a); r.m[2][2] = cosf(a);
        return r;
    }
    //LookAt 视图矩阵：设置摄像机位置和朝向
    static Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
        Vec3 f = (center - eye).norm();  //前向向量（看向目标的方向）
        Vec3 s = f.cross(up).norm();     //右向向量（垂直于前向和上方向）
        Vec3 u = s.cross(f);             //重新计算正交的上方向
        Mat4 r = identity();
        //设置旋转部分（基向量）
        r.m[0][0] = s.x;  r.m[1][0] = s.y;  r.m[2][0] = s.z;
        r.m[0][1] = u.x;  r.m[1][1] = u.y;  r.m[2][1] = u.z;
        r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
        //设置平移部分（将摄像机移动到原点）
        r.m[3][0] = -s.dot(eye);
        r.m[3][1] = -u.dot(eye);
        r.m[3][2] = f.dot(eye);
        return r;
    }
    //透视投影矩阵
    static Mat4 perspective(float fovY, float aspect, float zNear, float zFar) {
        float tanHalf = tanf(fovY * 0.5f);
        Mat4 r;
        r.m[0][0] = 1.f / (aspect * tanHalf);
        r.m[1][1] = 1.f / tanHalf;
        r.m[2][2] = -(zFar + zNear) / (zFar - zNear);       //深度先映射
        r.m[3][2] = -(2.f * zFar * zNear) / (zFar - zNear); //深度再偏移
        r.m[2][3] = -1.f;                                   //透视除法因子
        return r;
    }
};

//GPU端三角形
struct GPUTriangle {
    Vec3 v[3];//世界空间的三个顶点坐标
    Vec3 normal;//面法线
    unsigned char color[3];//RGB颜色值（0-255），该类型1字节，8位，256种
};

//渲染参数,从CPU传递给GPU，包含所有渲染所需的变换和光照参数(全局设置)
struct RenderParams {
    Mat4 mvp;       //模型-视图-投影 矩阵：模型空间到裁剪空间
    Mat4 modelView; //模型-视图矩阵：用于将法线变换到视图空间进行光照计算
    Vec3 lightDir;  //世界空间光照方向，指向光源
    Vec3 lightColor;//光源颜色
    Vec3 ambient;   //环境光颜色
    int  width;     //帧缓冲宽度（像素）
    int  height;    //帧缓冲高度（像素）
    int  numTris;   //三角形总数     
    int wireframe;  //非0为绘制线框模式
    float specPower;//高光指数（像镜面程度）
};

//一改：预计算三角形屏幕包围盒 + NDC 坐标
//所有数据都在GPU预处理阶段计算完成，光栅化阶段直接使用
struct TriCached {
    float nx[3], ny[3], nz[3];      //NDC 坐标（不需要裁剪的三角形直接使用）
    short minX, maxX, minY, maxY;   //屏幕像素包围盒（精确计算，不再是全屏幕）
    float baseR, baseG, baseB;      //预计算漫反射+环境光后的基础颜色（高光除外）
    float alpha;                    //透明度
    bool flipped;                   //原始三角形是否为背面（用于光照计算）
    float area;                     //原始三角形面积（不需要裁剪时使用）
    Vec3 normal;                    //预计算的面法线（已处理背面翻转）
    int clippedTriStart;            //裁剪后三角形在全局数组中的起始索引
    int clippedTriCount;            //裁剪后三角形数量（0=完全剔除，1=1个，2=2个）
};

//新增：裁剪后的三角形结构体
//所有与裁剪相关的计算都在预处理阶段完成，光栅化阶段直接使用
struct ClippedTri {
    float nx[3], ny[3], nz[3];      //裁剪后的NDC坐标（已完成透视除法）
    float area;                     //裁剪后的三角形面积（绝对值）
    bool flipped;                   //裁剪后的三角形是否为背面
};

//二改：屏幕分块：每个块存覆盖它的三角形索引列表
#define TILE_SIZE 16
#define MAX_TRIS_PER_TILE 40960     //每块最多 n 个三角形

struct TileData {
    int triIdx[MAX_TRIS_PER_TILE];  //三角形索引
    int triCount;                   //实际数量
};


//三改： BVH 节点结构（供 CPU 构建、GPU 遍历）
//说明：按模型空间 AABB 组织，叶子节点包含一段连续的三角形索引
struct BVHNode {
    float minX, minY, minZ;     //模型空间 AABB 最小角
    float maxX, maxY, maxZ;     //模型空间 AABB 最大角
    int left;                   //左子节点在数组中的索引，-1 表示当前为叶子节点
    int right;                  //右子节点索引，-1 表示叶子
    int triStart;               //叶子节点：在重排后三角形数组中的起始索引
    int triCount;               //叶子节点：包含的三角形数量（内部节点填 0）
};

//DLL引入API（实现在render.cu）
#ifdef BUILDING_CUDA_DLL    //如果是构建dll
# define CUDA_API __declspec(dllexport)//就定义为导出函数(微软dll导出表函数标记，标记函数后使dll内指定函数被exe可见)
#else                       //否则
# define CUDA_API           //定义为空
#endif

extern "C" {        //使用C语言链接规范（防止C++名称修饰，便于动态加载）

    //初始化CUDA：检测GPU，分配帧缓冲和深度缓冲内存。成功返回0，失败返回-1
    CUDA_API int cudaInit(
        int width, int height,      //输出图像分辨率
        char* outName,              //用于接收GPU名称的缓冲区
        int nameBufSize             //名称缓冲区大小
    );

    //三改：上传 BVH 节点数据到 GPU 显存（应在 cudaUploadTriangles 之后调用）
    CUDA_API void cudaUploadBVH(
        const BVHNode* nodes,       //主机端 BVH 节点数组指针（由 buildBVH 生成）
        int numNodes                //节点总数
    );

    //将三角形数据上传到GPU显存（可多次调用以重新上传）
    CUDA_API void cudaUploadTriangles(
        const GPUTriangle* tris,    //主机端三角形数组指针
        int numTris                 //三角形数量
    );

    //渲染一帧：执行光栅化和光照计算
    CUDA_API void cudaRender(
        const RenderParams& params, //渲染参数（MVP矩阵、光源等）
        unsigned int* g_hFramebuf   //输出缓冲区指针（BGRA格式，至少 width*height*4 字节）
    );

    //导出体素化后的三角形为OBJ文件
    extern "C" bool exportVoxelObj(
        const char* path,           //输出文件路径
        const GPUTriangle* tris,    //体素化后的三角形数组
        int numTris                 //三角形数量
    );

    //释放所有GPU资源：在卸载DLL之前调用
    CUDA_API void cudaShutdown(void);

}