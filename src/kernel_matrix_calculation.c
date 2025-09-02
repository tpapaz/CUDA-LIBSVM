#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macros
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(err) { \
    cublasStatus_t err_ = (err); \
    if (err_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Scalars
const float alpha = 1.0f;
const float beta = 0.0f;

void ckm(struct svm_problem *prob, struct svm_problem *pecm, float *gamma)
{
    double g_val = *gamma;
    int len_tv = prob->x[0].dim;
    int ntv = prob->l;

    float *h_tva = NULL, *h_dp = NULL;
    double *h_tv_sq = NULL, *h_v_f_g = NULL;
    float *d_tva = NULL, *d_vtm = NULL, *d_dot_prod = NULL;

    cublasHandle_t handle;

    // Allocate host memory
    h_tva = (float*)malloc(len_tv * ntv * sizeof(float));
    h_dp = (float*)malloc(ntv * sizeof(float));
    h_tv_sq = (double*)malloc(ntv * sizeof(double));
    h_v_f_g = (double*)malloc(ntv * sizeof(double));

    if (!h_tva || !h_dp || !h_tv_sq || !h_v_f_g) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Populate host data and compute squared norms
    for (int i = 0; i < ntv; ++i) {
        h_tv_sq[i] = 0;
        for (int j = 0; j < len_tv; ++j) {
            h_tva[i * len_tv + j] = (float)prob->x[i].values[j];
            h_tv_sq[i] += prob->x[i].values[j] * prob->x[i].values[j];
        }
    }

    // Initialize CUDA and CUBLAS
    CUDA_CHECK(cudaSetDevice(0));
    CUBLAS_CHECK(cublasCreate_v2(&handle));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_tva, len_tv * ntv * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_vtm, len_tv * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_dot_prod, ntv * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_tva, h_tva, len_tv * ntv * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < ntv; ++i) {
        // Copy current vector to device
        CUDA_CHECK(cudaMemcpy(d_vtm, &h_tva[i * len_tv], len_tv * sizeof(float), cudaMemcpyHostToDevice));

        // Perform matrix-vector multiplication: d_dot_prod = alpha * d_tva' * d_vtm + beta * d_dot_prod
        CUBLAS_CHECK(cublasSgemv_v2(handle, CUBLAS_OP_T, len_tv, ntv, &alpha, d_tva, len_tv, d_vtm, 1, &beta, d_dot_prod, 1));

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_dp, d_dot_prod, ntv * sizeof(float), cudaMemcpyDeviceToHost));

        // Compute RBF kernel
        for (int j = 0; j < ntv; ++j) {
            h_v_f_g[j] = exp(-g_val * (h_tv_sq[i] + h_tv_sq[j] - 2.0 * h_dp[j]));
        }

        pecm->x[i].values[0] = i + 1;
        for (int j = 0; j < ntv; ++j) {
            pecm->x[i].values[j + 1] = h_v_f_g[j];
        }
    }

    // Free memory
    free(h_tva);
    free(h_dp);
    free(h_tv_sq);
    free(h_v_f_g);
    CUDA_CHECK(cudaFree(d_tva));
    CUDA_CHECK(cudaFree(d_vtm));
    CUDA_CHECK(cudaFree(d_dot_prod));
    CUBLAS_CHECK(cublasDestroy_v2(handle));
}

void cal_km(struct svm_problem *p_km)
{
    float gamma = param.gamma;
    ckm(&prob, p_km, &gamma);
}
