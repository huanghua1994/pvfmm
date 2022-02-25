#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pvfmm.h>

// src_X: Size 3 * Ns, each row is a coordinate 
// trg_X: Size 3 * Nt, each row is a coordinate 
// src_V: Size Ns, source point values (input vector)
// trg_V: Size Nt, target point values (output vector)
void Laplace3D_potential(
    const double *src_X, const double *src_V, const int Ns,
    const double *trg_X, double *trg_V, const int Nt
)
{
    double oofp = 1.0 / (4.0 * M_PI);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nt; i++)
    {
        double val = 0.0;
        double trg_x = trg_X[3 * i + 0];
        double trg_y = trg_X[3 * i + 1];
        double trg_z = trg_X[3 * i + 2];
        #pragma omp simd
        for (int j = 0; j < Ns; j++)
        {
            double dx = src_X[3 * j + 0] - trg_x;
            double dy = src_X[3 * j + 1] - trg_y;
            double dz = src_X[3 * j + 2] - trg_z;
            double r2 = dx * dx + dy * dy + dz * dz;
            double rinv = (r2 > 0.0) ? (1.0 / sqrt(r2)) : 0.0;
            val += rinv * src_V[j];
        }
        trg_V[i] = val * oofp;
    }
}

void test_FMM_fin(void *ctx, const int num_point, const char *csv_fname) 
{
    int Ns = num_point, Nt = num_point;
    int krnl_dim = 1, pt_dim = 3;

    double *src_X  = (double *) malloc(Ns * pt_dim   * sizeof(double));
    double *src_V  = (double *) malloc(Ns * krnl_dim * sizeof(double));
    double *trg_X  = (double *) malloc(Nt * pt_dim   * sizeof(double));
    double *trg_V  = (double *) malloc(Nt * krnl_dim * sizeof(double));
    double *trg_V0 = (double *) malloc(Nt * krnl_dim * sizeof(double));

    // Read point coordinates from file
    FILE *inf = fopen(csv_fname, "r");
    for (int i = 0; i < num_point; i++)
    {
        for (int j = 0; j < pt_dim-1; j++)
            fscanf(inf, "%lf,", &src_X[pt_dim * i + j]);
        fscanf(inf, "%lf\n", &src_X[pt_dim * i + pt_dim-1]);
    }
    memcpy(trg_X, src_X, sizeof(double) * num_point * pt_dim);
    for (int i = 0; i < Ns * krnl_dim; i++) src_V[i] = drand48() - 0.5;
    printf("Set up coordinates done\n");

    // FMM evaluation
    double tt;
    int setup = 1;
    tt = -omp_get_wtime();
    PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, setup);
    printf("FMM evaluation time (with setup)    : %.3f s\n", tt + omp_get_wtime());

    // FMM evaluation (without setup)
    setup = 1;
    tt = -omp_get_wtime();
    PVFMMEvalD(src_X, src_V, NULL, Ns, trg_X, trg_V, Nt, ctx, setup);
    printf("FMM evaluation time (without setup) : %.3f s\n", tt + omp_get_wtime());

    // Direct evaluation
    int eval_Nt = 2000;
    tt = -omp_get_wtime();
    Laplace3D_potential(&src_X[0], &src_V[0], Ns, &trg_X[0], &trg_V0[0], eval_Nt);
    printf("Direct evaluation for %d target points time : %f\n", eval_Nt, tt + omp_get_wtime());

    double ref_2norm = 0.0, err_2norm = 0.0;
    for (int i = 0; i < eval_Nt; i++) 
    {
        double err = trg_V0[i] - trg_V[i];
        ref_2norm += trg_V0[i] * trg_V0[i];
        err_2norm += err * err;
    }
    ref_2norm = sqrt(ref_2norm);
    err_2norm = sqrt(err_2norm);
    printf("||trg_V0 - trg_V||_2 : %e\n", err_2norm / ref_2norm);

    free(src_X);
    free(src_V);
    free(trg_X);
    free(trg_V);
    free(trg_V0);
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    if (argc < 5)
    {
        printf("Usage: %s num_point max_leaf_size order csv_file\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int num_point = atoi(argv[1]);
    int max_leaf_size = atoi(argv[2]);
    int multipole_order = atoi(argv[3]);
    const char *csv_fname = argv[4];

    void *ctx;
    double box_size = -1;
    enum PVFMMKernel kernel = PVFMMLaplacePotential;
    double tt;
    tt = -omp_get_wtime();
    ctx = PVFMMCreateContextD(box_size, max_leaf_size, multipole_order, kernel, MPI_COMM_WORLD);
    printf("PVFMMCreateContextD time : %.3f s\n", tt + omp_get_wtime());
    test_FMM_fin(ctx, num_point, csv_fname);

    PVFMMPrintMyTimingsD(ctx);
    PVFMMDestroyContextD(&ctx);

    MPI_Finalize();
    return 0;
}
