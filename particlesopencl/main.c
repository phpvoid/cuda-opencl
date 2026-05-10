#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "ocl.h"

#define NUM_PARTICLES 65536
#define NUM_STEPS     100
#define DT            0.016f
#define GRAVITY       -9.81f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
} Particle;

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static void init_particles(Particle *p, int count) {
    for (int i = 0; i < count; i++) {
        p[i].x  = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        p[i].y  = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        p[i].z  = ((float)rand() / RAND_MAX) * 200.0f - 100.0f;
        p[i].vx = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        p[i].vy = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        p[i].vz = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
}

static double run_cpu(Particle *p, int count) {
    clock_t start = clock();
    for (int s = 0; s < NUM_STEPS; s++) {
        for (int i = 0; i < count; i++) {
            p[i].vy += GRAVITY * DT;
            p[i].x  += p[i].vx * DT;
            p[i].y  += p[i].vy * DT;
            p[i].z  += p[i].vz * DT;
        }
    }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

static void print_header(cl_device_id device) {
    char name[256] = {0}, vendor[256] = {0}, driver[256] = {0};
    cl_uint cu = 0;
    cl_ulong mem = 0;
    OCL cl;
    ocl_init(&cl);
    cl.GetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    cl.GetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    cl.GetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
    cl.GetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    cl.GetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
    printf("Device       : %s\n", name);
    printf("Vendor       : %s\n", vendor);
    printf("Driver       : %s\n", driver);
    printf("Compute Units: %u\n", cu);
    printf("Global Mem   : %llu MB\n", (unsigned long long)(mem / (1024*1024)));
    ocl_close(&cl);
}

int main(void) {
    OCL cl;
    cl_int err;
    size_t buf_sz = NUM_PARTICLES * sizeof(Particle);

    printf("=== Particle Simulator (CPU vs GPU Comparison) ===\n");
    printf("Particle Count: %d\n", NUM_PARTICLES);
    printf("Sim Steps     : %d\n\n", NUM_STEPS);

    err = ocl_init(&cl);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL init failed: %s\n", ocl_error_str(err));
        return 1;
    }

    cl_platform_id platform;
    cl.GetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    cl_uint num_devices;
    err = cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU found, trying CPU...\n");
        err = cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "No OpenCL device: %s\n", ocl_error_str(err));
            return 1;
        }
    }

    print_header(device);

    /* ---- CPU run ---- */
    Particle *cpu_p = malloc(buf_sz);
    srand(0);
    init_particles(cpu_p, NUM_PARTICLES);

    printf("\n--- CPU Simulation ---\n");
    double cpu_time = run_cpu(cpu_p, NUM_PARTICLES);
    printf("CPU time      : %.4f sec\n", cpu_time);
    printf("Particles/sec : %.0f\n", (NUM_PARTICLES * NUM_STEPS) / cpu_time);

    /* ---- GPU run (OpenCL) ---- */
    Particle *gpu_p = malloc(buf_sz);
    srand(0);
    init_particles(gpu_p, NUM_PARTICLES);

    cl_context ctx = cl.CreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = cl.CreateCommandQueue(ctx, device, 0, &err);

    char *src = read_file("particle_sim.cl");
    if (!src) { free(cpu_p); free(gpu_p); ocl_close(&cl); return 1; }
    cl_program prog = cl.CreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
    free(src);
    cl.BuildProgram(prog, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = cl.CreateKernel(prog, "simulate", &err);
    cl_mem d_particles = cl.CreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          buf_sz, gpu_p, &err);

    unsigned int count = NUM_PARTICLES;
    float dt = DT, gravity = GRAVITY;
    cl.SetKernelArg(kernel, 0, sizeof(cl_mem), &d_particles);
    cl.SetKernelArg(kernel, 1, sizeof(count),  &count);
    cl.SetKernelArg(kernel, 2, sizeof(dt),     &dt);
    cl.SetKernelArg(kernel, 3, sizeof(gravity),&gravity);

    size_t global_sz = NUM_PARTICLES;

    printf("\n--- GPU Simulation (OpenCL) ---\n");
    clock_t start = clock();
    for (int s = 0; s < NUM_STEPS; s++)
        cl.EnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_sz, NULL, 0, NULL, NULL);
    cl.Finish(queue);
    clock_t end = clock();
    double gpu_time = (double)(end - start) / CLOCKS_PER_SEC;

    cl.EnqueueReadBuffer(queue, d_particles, CL_TRUE, 0, buf_sz, gpu_p, 0, NULL, NULL);

    printf("GPU time      : %.4f sec\n", gpu_time);
    printf("Particles/sec : %.0f\n", (NUM_PARTICLES * NUM_STEPS) / gpu_time);

    /* ---- Comparison ---- */
    printf("\n=== Comparison ===\n");
    printf("%-18s %12s %16s\n", "", "CPU", "GPU (OpenCL)");
    printf("%-18s %12.4fs %12.4fs\n", "Time:", cpu_time, gpu_time);
    printf("%-18s %12.0f %16.0f\n", "Particles/sec:", (NUM_PARTICLES*NUM_STEPS)/cpu_time,
                                                       (NUM_PARTICLES*NUM_STEPS)/gpu_time);
    printf("%-18s %12s %12.1fx\n", "Speedup:", "-", cpu_time / gpu_time);

    int mismatches = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (fabsf(cpu_p[i].x - gpu_p[i].x) > 0.001f ||
            fabsf(cpu_p[i].y - gpu_p[i].y) > 0.001f ||
            fabsf(cpu_p[i].z - gpu_p[i].z) > 0.001f) {
            mismatches++;
        }
    }
    printf("%-18s %s\n", "Results match:", mismatches == 0 ? "YES" : "NO");
    if (mismatches > 0) printf("  mismatches: %d / %d\n", mismatches, NUM_PARTICLES);

    printf("\n--- Sample (GPU) ---\n");
    for (int i = 0; i < 4; i++)
        printf("  p[%d] pos=(%7.2f, %7.2f, %7.2f) vel=(%7.2f, %7.2f, %7.2f)\n",
               i, gpu_p[i].x, gpu_p[i].y, gpu_p[i].z,
               gpu_p[i].vx, gpu_p[i].vy, gpu_p[i].vz);

    free(cpu_p);
    free(gpu_p);
    cl.ReleaseMemObject(d_particles);
    cl.ReleaseKernel(kernel);
    cl.ReleaseProgram(prog);
    cl.ReleaseCommandQueue(queue);
    cl.ReleaseContext(ctx);
    ocl_close(&cl);

    return 0;
}
