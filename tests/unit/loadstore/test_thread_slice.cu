#include <gtest/gtest.h>

#include "src/loadstore/copy_gsm.cuh"

namespace {

template <int BlockM, int BlockN, int ThreadM, int ThreadK, int CopyK, int NWarps_>
struct TestKernelConfig {
    static constexpr int thread_m = ThreadM;
    static constexpr int thread_k = ThreadK;
    static constexpr int copy_k = CopyK;
    static constexpr int NWarps = NWarps_;
};

} // namespace

TEST(ThreadSliceShapeTest, OuterShapeIs4x2For16x128) {
    using Cfg = TestKernelConfig<16, 128, 4, 8, 8, 1>;
    using Map = loadstore::G2SLayoutThreadMap<Cfg, 16, 128, 8>;
    EXPECT_EQ(Map::IterRows, 4);
    EXPECT_EQ(Map::IterCols, 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
