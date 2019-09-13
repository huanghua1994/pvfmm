#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include <pvfmm_common.hpp>
#include <profile.hpp>
#include <fmm_pts.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>
#include <utils.hpp>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    int n_src = 0, n_trg = 0;
    if (argc >= 2) n_src = atoi(argv[1]);
    if (argc >= 3) n_trg = atoi(argv[2]);
    if (n_src == 0)
    {
        printf("n_src = ");
        scanf("%d", &n_src);
    }
    if (n_trg == 0)
    {
        printf("n_trg = ");
        scanf("%d", &n_trg);
    }
    
    const pvfmm::Kernel<double>* kernel;
    kernel = &pvfmm::LaplaceKernel<double>::potential();
    
    double *src_coord = (double*) malloc(sizeof(double) * n_src * 3);
    double *trg_coord = (double*) malloc(sizeof(double) * n_trg * 3);
    double *src_val   = (double*) malloc(sizeof(double) * n_src);
    double *trg_val   = (double*) malloc(sizeof(double) * n_src);
    
    for (int i = 0; i < n_src * 3; i++) src_coord[i] = drand48();
    for (int i = 0; i < n_trg * 3; i++) trg_coord[i] = drand48();
    for (int i = 0; i < n_src; i++) src_val[i] = drand48();
    
    pvfmm::Profile::Enable(true);
    int nthread = omp_get_max_threads();
    for (int k = 0; k < 5; k++)
    {
        pvfmm::Profile::Tic("Direct N-Body", &comm, false, 1);
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            int trg_sidx = tid * n_trg / nthread;
            int trg_eidx = (tid + 1) * n_trg / nthread;
            int n_trg_thread = trg_eidx - trg_sidx;
            kernel->ker_poten(
                src_coord, n_src, src_val, 1, 
                trg_coord, n_trg_thread, trg_val + trg_sidx, NULL
            );
        }
        pvfmm::Profile::Toc();
    }
    pvfmm::Profile::print(&comm);
    
    free(src_coord);
    free(trg_coord);
    free(src_val);
    free(trg_val);
    
    MPI_Finalize();
    return 0;
}