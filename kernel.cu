
#include "ADAM_optimiser.h"
#include "printstring_helper.h"
#include <stdio.h>
#include <vector>
#include <chrono>

void generate_polynomial(const smart_cpu_buffer<float>& coefficients, smart_cpu_buffer<float>& output)
{
    for (int i = 0; i < output.dedicated_len; i++)
    {
        float sum = 0.f, x = 1.f;
        for (int j = 0; j < coefficients.dedicated_len; j++)
        {
            sum += x * coefficients[j];
            x *= i / (float)output.dedicated_len * 3.f - 1.5f;
        }
        output[i] = sum;
    }
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    rng_state state = rng_state(std::chrono::steady_clock().now().time_since_epoch().count() >> 20);
    
    neural_net<1u> net({ 250, 50, 5 }, state);
    ADAMax<1u> optimiser = ADAMax<1u>(net);
    smart_cpu_buffer<float> input(net.input_size());
    smart_cpu_buffer<float> ground_truth_and_target_change(net.output_size());

    // Example test: finding coefficients of approximating polynomial.
    for (int i = 0; i < 100; i++)
    {
        float loss = 0.f;
        for (int j = 0; j < 200; j++)
        {
            for (int k = 0; k < ground_truth_and_target_change.dedicated_len; k++)
                ground_truth_and_target_change[k] = state.gen_float(i * 1000 + j * 10 + k);
            generate_polynomial(ground_truth_and_target_change, input);
            net.evaluate_with_ext_input(input);
            for (int k = 0; k < ground_truth_and_target_change.dedicated_len; k++)
            {
                ground_truth_and_target_change[k] -= net.output[k];
                loss += ground_truth_and_target_change[k] * ground_truth_and_target_change[k];
            }
            optimiser.apply_adaMax(ground_truth_and_target_change, 0.98f, 0.98f, 0.001f);
        }
        writeline_t(loss * .005f);
    }

    while (true)
        _sleep(1000);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
