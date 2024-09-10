
#include "ADAM_optimiser.h"
#include "printstring_helper.h"
#include <stdio.h>
#include <vector>
#include <chrono>

constexpr uint epoch_size = 500u;

void generate_noisy_polynomial(const smart_cpu_buffer<float>& coefficients, smart_cpu_buffer<float>& output, rng_state& state)
{
    for (int i = 0; i < output.dedicated_len; i++)
    {
        float sum = 0.f, x = 1.f;
        for (int j = 0; j < coefficients.dedicated_len; j++)
        {
            sum += x * coefficients[j];
            x *= i / (float)output.dedicated_len * 3.f - 1.5f;
        }
        output[i] = sum + state.gen_float() * .25f;
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
    for (uint i = 0; i < 50; i++)
    {
        float loss = 0.f; float f = 1.f - .5f / (i + 10);
        for (uint j = 0; j < epoch_size; j++)
        {
            for (int k = 0; k < ground_truth_and_target_change.dedicated_len; k++)
                ground_truth_and_target_change[k] = state.gen_float(i * 1000 + j * 10 + k);
            generate_noisy_polynomial(ground_truth_and_target_change, input, state);
            net.evaluate_with_ext_input(input);

            if (j + 1u == epoch_size)
            {
                writeline("Sample:\nGround Truth: " + print_buffer(ground_truth_and_target_change));
                writeline("Prediction: " + print_buffer(net.output));
            }

            for (int k = 0; k < ground_truth_and_target_change.dedicated_len; k++)
            {
                ground_truth_and_target_change[k] -= net.output[k];
                loss += ground_truth_and_target_change[k] * ground_truth_and_target_change[k];
            }
            optimiser.apply_adaMax(ground_truth_and_target_change, f, f, 0.0015f, 1E-5f);
        }
        writeline("Average Loss: " + std::to_string(loss / epoch_size) + "\n");
        _sleep(1000);
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
