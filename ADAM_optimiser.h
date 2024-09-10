#ifndef ADAM_H
#define ADAM_H

#include "helper_math.h"
#include "CUDA_memory.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <cassert>

struct rng_state
{
	int4 state;
	__host__ __device__ rng_state(int a = -1, int b = -2, int c = -3, int d = -4) : state(make_int4(a,b,c,d)) { }
	__host__ __device__ void update_state(int s = 0, int iters = 3)
	{
		for (int i = 0; i < iters; i++)
		{
			int t = state.x ^ (s * 21481264 + i - 118947813) ^ (state.w << 3);
			t ^= t >> 5; t += state.y * (1481912741 ^ state.z);
			state.w = state.z; state.z = state.y; state.y = state.x;
			state.x = t ^ (t << 3);
		}
	}
	
	__host__ __device__ int gen_int(int s = 0)
	{
		update_state(s, 3);
		return state.x;
	}
	__host__ __device__ int2 gen_int2(int s = 0)
	{
		update_state(s, 3);
		int v1 = state.x;
		update_state(s, 2);
		int v2 = state.x;
		return make_int2(v1, v2);
	}
	__host__ __device__ int3 gen_int3(int s = 0)
	{
		update_state(s, 7);
		return make_int3(state.x, state.y, state.z);
	}
	__host__ __device__ int4 gen_int4(int s = 0)
	{
		update_state(s, 8);
		int v1 = state.x;
		int v2 = state.y;
		int v3 = state.z;
		update_state(s, 2);
		return make_int4(v1,v2,v3,state.x);
	}

	__host__ __device__ int gen_int(int min_incl, int max_excl, int s = 0)
	{
		update_state(s, 3);
		return (int)(((uint)state.x) % ((uint)(max_excl - min_incl))) + min_incl;
	}
	__host__ __device__ float gen_float(int s = 0)
	{
		return gen_int(s) / 2147483648.f;
	}
	__host__ __device__ float2 gen_float2(int s = 0)
	{
		return make_float2(gen_int2(s)) / 2147483648.f;
	}
	__host__ __device__ float3 gen_float3(int s = 0)
	{
		return make_float3(gen_int3(s)) / 2147483648.f;
	}
	__host__ __device__ float4 gen_float4(int s = 0)
	{
		return make_float4(gen_int4(s)) / 2147483648.f;
	}
};

__device__ constexpr float leaky_param = 0.0625f;

__host__ __device__ float leaky_ReLU_grad(float var)
{
	return var < 0.f ? leaky_param : 1.f;
}
__host__ __device__ float leaky_ReLU_activation(float in, const float bias)
{
	in += bias;
	return in < 0.f ? in * leaky_param : in;
}

__host__ __device__ float squareplus_grad(float var)
{
	var = ((1.f + leaky_param) * var - sqrtf((1.f - leaky_param) * (1.f - leaky_param) * (leaky_param + var * var))) * (.5f / leaky_param);
	return 0.5f * (1.f + leaky_param) + (0.5f * (1.f - leaky_param)) * var * rsqrtf(1.f + var * var);
}
__host__ __device__ float squareplus_activation(float in, const float bias)
{
	in += bias;
	return 0.5f * ((1.f - leaky_param) * sqrtf(in * in + 1.f) + (1.f + leaky_param) * in);
}

// sigma' compose sigma^-1
__host__ __device__ float activation_grad_inv(float var)
{
	return squareplus_grad(var);
}
// sigma
__host__ __device__ float activation_function(float in, const float bias)
{
	return squareplus_activation(in, bias);
}

__host__ __device__ void f2sum(float& sum, float& err, const float add)
{
	volatile float s = sum + add;
	volatile float z = s - sum;

	sum = s;
	err = add - z;
}
__global__ void hidden_layer_invoke(const float* mat, const float* invec, float* outvec, const float* bias, const uint old_layer_size, const uint new_layer_size)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= new_layer_size)
		return;

	float s = 0.f, e = 0.f;
	for (uint i = 0; i < old_layer_size; i++)
		f2sum(s, e, invec[i] * mat[i * new_layer_size + idx]);

	outvec[idx] = activation_function(s + e, bias[idx]);
}
__global__ void init_const(float* arr, const float v, const uint len)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= len) { return; }

	arr[idx] = v;
}
__global__ void generate_rng(float* arr, const float scale, const uint len, rng_state init_state)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= len) { return; }

	arr[idx] = init_state.gen_float(idx) * scale;
}

void set_random_buffer(smart_gpu_buffer<float>& buffer, const float scale, rng_state& rng)
{
	dim3 threads(buffer.dedicated_len < 256u ? buffer.dedicated_len : 256u);
	dim3 blocks((uint)ceilf(buffer.dedicated_len / (float)threads.x));
	generate_rng<<<blocks, threads>>>(buffer.gpu_buffer_ptr, scale, buffer.dedicated_len, rng);
	rng.update_state(buffer.dedicated_len * 26417612 + 37481783, 10);
}
void set_uniform(smart_gpu_buffer<float>& buffer, const float v)
{
	dim3 threads(buffer.dedicated_len < 256u ? buffer.dedicated_len : 256u);
	dim3 blocks((uint)ceilf(buffer.dedicated_len / (float)threads.x));
	init_const<<<blocks, threads>>> (buffer.gpu_buffer_ptr, v, buffer.dedicated_len);
}

struct hidden_layer
{
	smart_gpu_buffer<float> weight;
	smart_gpu_buffer<float> bias;
	uint old_layer_size, new_layer_size;

	void generate_random(rng_state& rng)
	{
		set_random_buffer(weight, rsqrtf((old_layer_size + new_layer_size) * .166666666666666f), rng);
		set_random_buffer(bias, 1.f, rng);
	}
	void blank_slate()
	{
		set_uniform(weight, 0.f);
		set_uniform(bias, 0.f);
	}

	hidden_layer() : weight(), old_layer_size(0u), new_layer_size(0u) {}
	hidden_layer(uint old_layer_size, uint new_layer_size) : old_layer_size(old_layer_size), new_layer_size(new_layer_size), weight((size_t)new_layer_size* old_layer_size), bias(new_layer_size) { assert(old_layer_size * new_layer_size > 0); }
	bool apply(const smart_gpu_buffer<float>& invec, smart_gpu_buffer<float>& outvec)
	{
		if (invec.dedicated_len != old_layer_size || outvec.dedicated_len != new_layer_size || !weight.created)
			return false;
		
		dim3 threads(new_layer_size < 256u ? new_layer_size : 256u);
		dim3 blocks((uint)ceilf(new_layer_size / (float)threads.x));
		hidden_layer_invoke<<<blocks, threads>>>(weight.gpu_buffer_ptr, invec.gpu_buffer_ptr, outvec.gpu_buffer_ptr, bias.gpu_buffer_ptr, old_layer_size, new_layer_size);
		return true;
	}
	void destroy()
	{
		weight.destroy();
		bias.destroy();
	}
};

template <uint hidden_layers>
struct neural_net
{
	smart_gpu_cpu_buffer<float> output;
	hidden_layer hidden[hidden_layers + 1u];
	smart_gpu_buffer<float> hidden_values[hidden_layers + 1u];

	uint input_size() const {
		return hidden_values[0].dedicated_len;
	}
	uint output_size() const {
		return output.dedicated_len;
	}

	void destroy()
	{
		output.destroy();
		for (int i = 0; i < hidden_layers + 1u; i++)
		{
			hidden_values[i].destroy();
			hidden[i].destroy();
		}
	}
	void evaluate()
	{
		for (int i = 0; i < hidden_layers; i++)
			hidden[i].apply(hidden_values[i], hidden_values[i + 1]);
		hidden[hidden_layers].apply(hidden_values[hidden_layers], output);
		output.copy_to_cpu();
	}
	void evaluate_with_ext_input(const smart_cpu_buffer<float>& input)
	{
		copy_to_gpu(hidden_values[0], input);
		evaluate();
	}

	neural_net(const std::vector<uint>& node_count, rng_state& state, bool init_random = true)
	{
		assert(node_count.size() == (hidden_layers + 2u));
		for (int i = 0; i < hidden_layers + 1u; i++)
		{
			hidden_values[i] = smart_gpu_buffer<float>(node_count[i]);
			hidden[i] = hidden_layer(node_count[i], node_count[i + 1u]);
			if (init_random)
				hidden[i].generate_random(state);
			else 
				hidden[i].blank_slate();
		}
		output = smart_gpu_cpu_buffer<float>(node_count[hidden_layers + 1u]);
	}
};

__global__ void backpropagation(float* weight_grad, float* bias_grad, float* node_grad, const float* next_node_grad,
								const float* weight, const float* curr_node, const float* next_node, 
								const uint curr_size, const uint next_size)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < next_size)
	{
		float new_grad = next_node_grad[idx] * activation_grad_inv(next_node[idx]); // Could be cached but how? Shared memory not guaranteed

		bias_grad[idx] = new_grad;
		for (int i = 0; i < curr_size; i++)
			weight_grad[i * next_size + idx] = new_grad * curr_node[i];
	}
	if (idx >= curr_size)
		return;

	float s1 = 0.f, e1 = 0.f, s2 = 1.f, e2 = 0.f;
	for (uint i = 0; i < next_size; i++) // to check
	{
		float w = weight[i + idx * next_size];
		float g = activation_grad_inv(next_node[i]); // Could be reused but how? Shared memory not guaranteed
		f2sum(s1, e1, next_node_grad[i] * g * w);
		f2sum(s2, e2, w * w * g);
	}
	node_grad[idx] = (s1 + e1) / (s2 + e2);
}

__global__ void apply_adaMax_single_layer(const float* weight_grad, const float* bias_grad,
										float* weight_max, float* bias_max, 
										float* weight_momentum, float* bias_momentum, 
										float* weight, float* bias, const float l1_regularization,
										const float momentum, const float adaptive_decay_rate, const float learn_rate,
										const uint curr_size, const uint next_size)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= curr_size * next_size)
		return;

	if (idx < next_size)
	{
		float bias_gr = bias_grad[idx];
		float bias_mom = lerp(bias_momentum[idx], bias_gr, 1.f - momentum);
		float bias_max_temp = fmaxf(bias_max[idx] * adaptive_decay_rate, fabsf(bias_gr));
		float bias_fin = bias[idx] + bias_mom / (bias_max_temp + 1E-8f) * learn_rate;

		bias_momentum[idx] = bias_mom;
		bias_max[idx] = bias_max_temp;
		bias[idx] = isnan(bias_fin) ? 0.f : bias_fin; // neuron dies and is replaced ifnan
	}

	float weight_gr = weight_grad[idx];
	float weight_mom = lerp(weight_momentum[idx], weight_gr, 1.f - momentum);
	float weight_max_temp = fmaxf(weight_max[idx] * adaptive_decay_rate, fabsf(weight_gr));
	float weight_fin = weight[idx] + weight_mom / (weight_max_temp + 1E-8f) * learn_rate;

	weight_momentum[idx] = weight_mom;
	weight_max[idx] = weight_max_temp;
	weight[idx] = isnan(weight_fin) ? rsqrtf(curr_size) : weight_fin - sign(weight_fin) * fminf(l1_regularization, fabsf(weight_fin)); // synapse dies and is replaced ifnan
}

template <uint hidden_layers>
struct ADAMax
{
	neural_net<hidden_layers>& model;

	smart_gpu_buffer<float> weight_grad[hidden_layers + 1u];
	smart_gpu_buffer<float> weight_max[hidden_layers + 1u];
	smart_gpu_buffer<float> weight_momentum[hidden_layers + 1u];
	smart_gpu_buffer<float> bias_grad[hidden_layers + 1u];
	smart_gpu_buffer<float> bias_max[hidden_layers + 1u];
	smart_gpu_buffer<float> bias_momentum[hidden_layers + 1u];

	smart_gpu_buffer<float> node_grad[hidden_layers + 2u];

private:
	void backpropagation_single_layer(smart_gpu_buffer<float>& weight_grad, smart_gpu_buffer<float>& bias_grad, smart_gpu_buffer<float>& node_grad, 
		const smart_gpu_buffer<float>& next_node_grad, const smart_gpu_buffer<float> weight, const smart_gpu_buffer<float>& curr_node, const smart_gpu_buffer<float>& next_node)
	{
		uint process_len = curr_node.dedicated_len < next_node.dedicated_len ? next_node.dedicated_len : curr_node.dedicated_len;
		dim3 threads(process_len < 256u ? process_len : 256u);
		dim3 blocks((uint)ceilf(process_len / (float)threads.x));
		backpropagation<<<blocks,threads>>>(weight_grad.gpu_buffer_ptr, bias_grad.gpu_buffer_ptr, node_grad.gpu_buffer_ptr, next_node_grad.gpu_buffer_ptr,
			weight.gpu_buffer_ptr, curr_node.gpu_buffer_ptr, next_node.gpu_buffer_ptr, curr_node.dedicated_len, next_node.dedicated_len);
	}
	void apply_adaMax_layer(const uint layer, const float momentum, const float adaptive_decay_rate, const float learn_rate, const float regularization)
	{
		uint size = weight_max[layer].dedicated_len;
		dim3 threads(size < 256u ? size : 256u);
		dim3 blocks((uint)ceilf(size / (float)threads.x));
		
		apply_adaMax_single_layer<<<blocks, threads>>>(weight_grad[layer], bias_grad[layer],
			weight_max[layer], bias_max[layer], weight_momentum[layer], bias_momentum[layer],
			model.hidden[layer].weight, model.hidden[layer].bias, regularization, momentum, adaptive_decay_rate, learn_rate,
			model.hidden[layer].old_layer_size, model.hidden[layer].new_layer_size);
	}

public:
	/// <summary>
	/// Computes the required change in weights, biases and node activations in order to change the output in the desired direction
	/// </summary>
	/// <param name="target_change">Desired change</param>
	void backpropagation_full(smart_cpu_buffer<float>& target_change)
	{
		copy_to_gpu(node_grad[hidden_layers + 1u], target_change);
		backpropagation_single_layer(weight_grad[hidden_layers], bias_grad[hidden_layers], node_grad[hidden_layers], node_grad[hidden_layers + 1u], model.hidden[hidden_layers].weight, model.hidden_values[hidden_layers], model.output);
		if (hidden_layers > 0)
			for (int i = hidden_layers - 1; i >= 0; i--)
				backpropagation_single_layer(weight_grad[i], bias_grad[i], node_grad[i], node_grad[i + 1u], model.hidden[i].weight, model.hidden_values[i], model.hidden_values[i + 1u]);
	}
	/// <summary>
	/// Optimise for the desired change using the ADAM optimisation method.
	/// </summary>
	/// <param name="target_change">Desired change</param>
	/// <param name="momentum">Momentum serving to smooth and regulate changes.</param>
	/// <param name="adaptive_decay_rate">Decay rate for learning rate correction.</param>
	/// <param name="learn_rate">Rate at which changes are applied.</param>
	void apply_adaMax(smart_cpu_buffer<float>& target_change, const float momentum, const float adaptive_decay_rate, const float learn_rate, const float l1_regularization_strength)
	{
		backpropagation_full(target_change);
		for (uint i = 0; i < hidden_layers + 1u; i++)
			apply_adaMax_layer(i, momentum, adaptive_decay_rate, learn_rate, l1_regularization_strength);
	}
	ADAMax(neural_net<hidden_layers>& model) : model(model)
	{
		node_grad[hidden_layers + 1u] = smart_gpu_buffer<float>(model.output.dedicated_len);
		for (int i = 0; i < hidden_layers + 1u; i++)
		{
			node_grad[i] = smart_gpu_buffer<float>(model.hidden_values[i].dedicated_len);
			weight_grad[i] = smart_gpu_buffer<float>(model.hidden[i].weight.dedicated_len);
			weight_max[i] = smart_gpu_buffer<float>(model.hidden[i].weight.dedicated_len);
			weight_momentum[i] = smart_gpu_buffer<float>(model.hidden[i].weight.dedicated_len);
			bias_grad[i] = smart_gpu_buffer<float>(model.hidden[i].bias.dedicated_len);
			bias_max[i] = smart_gpu_buffer<float>(model.hidden[i].bias.dedicated_len);
			bias_momentum[i] = smart_gpu_buffer<float>(model.hidden[i].bias.dedicated_len);

			set_uniform(weight_max[i], .01f);
			set_uniform(bias_max[i], .01f);
			set_uniform(weight_momentum[i], 0.f);
			set_uniform(bias_momentum[i], 0.f);
		}
	}
};

// Untested, experimental

template <uint hidden_layers>
static int sample_gibbs_dist(neural_net< hidden_layers>& model, smart_cpu_buffer<float>& inout, rng_state& state)
{
	model.evaluate_with_ext_input(inout);

	float t = 1.f;
	for (uint j = 0; j < model.output.dedicated_len; j++)
	{
		float s = expf(model.output[j]);
		inout[j] = s; t += s;
	}
	for (uint j = 0; j < model.output.dedicated_len; j++)
		inout[j] /= t;

	t = (state.gen_float() * .5f + .5f) - (1.f / t);
	for (uint j = 0; j < model.output.dedicated_len; j++)
	{
		if (t < 0.f) { return j; }
		t -= inout[j];
	}
	return model.output.dedicated_len;
}

template <uint hidden_layers>
static void optimise_reinforcement_learning(ADAMax<hidden_layers>& optimiser, std::vector<float>& reinforce_in, std::vector<float>& discourage_in, smart_cpu_buffer<float>& temp,
											const float momentum, const float adaptive_decay_rate, const float learn_rate, const float regularization, const float action_weight, rng_state& state)
{
	const uint in_size = optimiser.model.hidden_values[0].dedicated_len;
	const uint out_size = optimiser.model.output.dedicated_len;
	assert(reinforce_in.size() % in_size == 0 && discourage_in.size() % in_size == 0 && temp.dedicated_len == in_size);
	for (uint i = 0, s = reinforce_in.size() / in_size; i < s; i++)
	{
		for (uint j = 0; j < in_size; j++)
			temp[j] = reinforce_in[i * in_size + j];

		uint sample = sample_gibbs_dist<hidden_layers>(optimiser.model, temp, state);
		for (uint j = 0; j < out_size; j++)
			temp[j] = lerp((j + 1u == sample) ? 1.f : -1.f / out_size, temp[j] - 1.f / (out_size + 1.f), action_weight);
		optimiser.apply_adaMax(temp, momentum, adaptive_decay_rate, learn_rate, 0.f);
	}
	for (uint i = 0, s = discourage_in.size() / in_size; i < s; i++)
	{
		for (uint j = 0; j < in_size; j++)
			temp[j] = discourage_in[i * in_size + j];

		uint sample = sample_gibbs_dist<hidden_layers>(optimiser.model, temp, state);
		for (uint j = 0; j < out_size; j++)
			temp[j] = lerp((j + 1u == sample) ? -1.f : 1.f / out_size, 1.f / (out_size + 1.f) - temp[j], action_weight);
		optimiser.apply_adaMax(temp, momentum, adaptive_decay_rate, learn_rate, (i + 1u == s) ? regularization : 0.f);
	}
}

#endif