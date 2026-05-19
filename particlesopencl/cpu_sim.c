#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_PARTICLES 65536
#define NUM_STEPS 100
#define DT 0.016f
#define GRAVITY -9.81f

typedef struct {
    float x, y, z;
    float vx, vy, vz;
} Particle;

static void simulate_cpu(Particle *p, int count, int steps, float dt, float gravity) {
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < count; i++) {
            p[i].vy += gravity * dt;
            p[i].x  += p[i].vx * dt;
            p[i].y  += p[i].vy * dt;
            p[i].z  += p[i].vz * dt;
        }
    }
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

int main(void) {
    Particle *particles = malloc(NUM_PARTICLES * sizeof(Particle));
    srand(0);
    init_particles(particles, NUM_PARTICLES);

    printf("=== Particle Simulator (CPU) ===\n");
    printf("Particle Count: %d\n", NUM_PARTICLES);
    printf("Steps: %d\n", NUM_STEPS);

    clock_t start = clock();
    simulate_cpu(particles, NUM_PARTICLES, NUM_STEPS, DT, GRAVITY);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU time: %.4f seconds\n", elapsed);
    printf("Particles/sec: %.0f\n", (NUM_PARTICLES * NUM_STEPS) / elapsed);

    printf("\n Sample particles \n");
    for (int i = 0; i < 5; i++) {
        printf("  p[%d] pos=(%8.2f, %8.2f, %8.2f) vel=(%8.2f, %8.2f, %8.2f)\n",
               i, particles[i].x, particles[i].y, particles[i].z,
               particles[i].vx, particles[i].vy, particles[i].vz);
    }
    printf("\n");

    free(particles);
    return 0;
}
