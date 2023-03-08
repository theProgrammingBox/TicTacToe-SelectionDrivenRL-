#pragma once
#include "Header.h"

class NeuralNetwork
{
public:
	static constexpr uint32_t BOARD_WIDTH = 3;
	static constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;

	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE + 1;
	static constexpr uint32_t LEAKY_ONE_SIZE = 18;
	static constexpr uint32_t WEIGHT_ONE_SIZE = INPUT_SIZE * LEAKY_ONE_SIZE;
	static constexpr uint32_t LEAKY_TWO_SIZE = 18;
	static constexpr uint32_t WEIGHT_TWO_SIZE = LEAKY_ONE_SIZE * LEAKY_TWO_SIZE;
	static constexpr uint32_t SOFTMAX_SIZE = 9;
	static constexpr uint32_t WEIGHT_THREE_SIZE = LEAKY_TWO_SIZE * SOFTMAX_SIZE;

	struct Parameters
	{
		float weightMatrixOne[WEIGHT_ONE_SIZE];
		float biasMatrixOne[LEAKY_ONE_SIZE];
		float weightMatrixTwo[WEIGHT_TWO_SIZE];
		float weightMatrixThree[WEIGHT_THREE_SIZE];

		float weightMatrixThreeDerivative[WEIGHT_THREE_SIZE];
		float weightMatrixTwoDerivative[WEIGHT_TWO_SIZE];
		float biasMatrixOneDerivative[LEAKY_ONE_SIZE];
		float weightMatrixOneDerivative[WEIGHT_ONE_SIZE];

		Parameters()
		{
			/*cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)), sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)));
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -sqrt(6.0f / (LEAKY_ONE_SIZE + LEAKY_TWO_SIZE)), sqrt(6.0f / (LEAKY_ONE_SIZE + LEAKY_TWO_SIZE)));
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -sqrt(6.0f / (LEAKY_TWO_SIZE + SOFTMAX_SIZE)), sqrt(6.0f / (LEAKY_TWO_SIZE + SOFTMAX_SIZE)));*/
			/*cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -0.01f, 0.01f);
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -0.01f, 0.01f);
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -0.01f, 0.01f);*/
			cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -1.0f, 1.0f);
			memset(biasMatrixOne, 0, sizeof(float) * LEAKY_ONE_SIZE);
			
			memset(weightMatrixThreeDerivative, 0, sizeof(float) * WEIGHT_THREE_SIZE);
			memset(weightMatrixTwoDerivative, 0, sizeof(float) * WEIGHT_TWO_SIZE);
			memset(biasMatrixOneDerivative, 0, sizeof(float) * LEAKY_ONE_SIZE);
			memset(weightMatrixOneDerivative, 0, sizeof(float) * WEIGHT_ONE_SIZE);
		}

		void Update(float learningRate)
		{
			cpuClip(weightMatrixThreeDerivative, WEIGHT_THREE_SIZE, learningRate, -0.01f, 0.01f);
			cpuClip(weightMatrixTwoDerivative, WEIGHT_TWO_SIZE, learningRate, -0.01f, 0.01f);
			cpuClip(biasMatrixOneDerivative, LEAKY_ONE_SIZE, learningRate, -0.01f, 0.01f);
			cpuClip(weightMatrixOneDerivative, WEIGHT_ONE_SIZE, learningRate, -0.01f, 0.01f);
			
			cpuSaxpy(WEIGHT_THREE_SIZE, &GLOBAL::ONEF, weightMatrixThreeDerivative, 1, weightMatrixThree, 1);
			cpuSaxpy(WEIGHT_TWO_SIZE, &GLOBAL::ONEF, weightMatrixTwoDerivative, 1, weightMatrixTwo, 1);
			cpuSaxpy(LEAKY_ONE_SIZE, &GLOBAL::ONEF, biasMatrixOneDerivative, 1, biasMatrixOne, 1);
			cpuSaxpy(WEIGHT_ONE_SIZE, &GLOBAL::ONEF, weightMatrixOneDerivative, 1, weightMatrixOne, 1);
			
			memset(weightMatrixThreeDerivative, 0, sizeof(float) * WEIGHT_THREE_SIZE);
			memset(weightMatrixTwoDerivative, 0, sizeof(float) * WEIGHT_TWO_SIZE);
			memset(biasMatrixOneDerivative, 0, sizeof(float) * LEAKY_ONE_SIZE);
			memset(weightMatrixOneDerivative, 0, sizeof(float) * WEIGHT_ONE_SIZE);
		}
	};

	struct Computation
	{
		Parameters* parameters;
		float inputMatrix[INPUT_SIZE];
		float productMatrixOne[LEAKY_ONE_SIZE];
		float leakyMatrixOne[LEAKY_ONE_SIZE];
		float productMatrixTwo[LEAKY_TWO_SIZE];
		float leakyMatrixTwo[LEAKY_TWO_SIZE];
		float productMatrixThree[SOFTMAX_SIZE];
		float softmaxMatrix[SOFTMAX_SIZE];
		uint32_t sampledAction;
		bool* isWinner;

		uint32_t ForwardPropagate(Parameters* parameters, float* board, float turn, bool* isWinner)
		{
			this->parameters = parameters;
			this->isWinner = isWinner;

			memcpy(inputMatrix, board, sizeof(int) * BOARD_SIZE);
			inputMatrix[BOARD_SIZE] = turn;
			cpuSgemmStridedBatched(
				false, false,
				LEAKY_ONE_SIZE, 1, INPUT_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixOne, LEAKY_ONE_SIZE, 0,
				inputMatrix, INPUT_SIZE, 0,
				&GLOBAL::ZEROF,
				productMatrixOne, LEAKY_ONE_SIZE, 0,
				1);
			cpuSaxpy(LEAKY_ONE_SIZE, &GLOBAL::ONEF, parameters->biasMatrixOne, 1, productMatrixOne, 1);
			cpuLeakyRelu(productMatrixOne, leakyMatrixOne, LEAKY_ONE_SIZE);
			cpuSgemmStridedBatched(
				false, false,
				LEAKY_TWO_SIZE, 1, LEAKY_ONE_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixTwo, LEAKY_TWO_SIZE, 0,
				leakyMatrixOne, LEAKY_ONE_SIZE, 0,
				&GLOBAL::ZEROF,
				productMatrixTwo, LEAKY_TWO_SIZE, 0,
				1);
			cpuLeakyRelu(productMatrixTwo, leakyMatrixTwo, LEAKY_TWO_SIZE);
			cpuSgemmStridedBatched(
				false, false,
				SOFTMAX_SIZE, 1, LEAKY_TWO_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixThree, SOFTMAX_SIZE, 0,
				leakyMatrixTwo, LEAKY_TWO_SIZE, 0,
				&GLOBAL::ZEROF,
				productMatrixThree, SOFTMAX_SIZE, 0,
				1);
			cpuSoftmax(productMatrixThree, softmaxMatrix, SOFTMAX_SIZE);

			float randomNumber = GLOBAL::RANDOM.Rfloat();
			sampledAction = 0;
			for (;;)
			{
				randomNumber -= softmaxMatrix[sampledAction];
				if (randomNumber <= 0)
					break;
				sampledAction -= (++sampledAction >= SOFTMAX_SIZE) * SOFTMAX_SIZE;
			}
			return sampledAction;
		}

		void BackPropagate()
		{
			float productMatrixThreeDerivative[SOFTMAX_SIZE];
			float leakyMatrixTwoDerivative[LEAKY_TWO_SIZE];
			float productMatrixTwoDerivative[LEAKY_TWO_SIZE];
			float leakyMatrixOneDerivative[LEAKY_ONE_SIZE];
			float productMatrixOneDerivative[LEAKY_ONE_SIZE];

			memset(productMatrixThreeDerivative, 0, sizeof(float) * SOFTMAX_SIZE);
			memset(leakyMatrixTwoDerivative, 0, sizeof(float) * LEAKY_TWO_SIZE);
			memset(productMatrixTwoDerivative, 0, sizeof(float) * LEAKY_TWO_SIZE);
			memset(leakyMatrixOneDerivative, 0, sizeof(float) * LEAKY_ONE_SIZE);
			memset(productMatrixOneDerivative, 0, sizeof(float) * LEAKY_ONE_SIZE);
			
			cpuSoftmaxDerivative(productMatrixThree, productMatrixThreeDerivative, *isWinner, sampledAction, SOFTMAX_SIZE);
			cpuSgemmStridedBatched(
				false, true,
				SOFTMAX_SIZE, LEAKY_TWO_SIZE, 1,
				&GLOBAL::ONEF,
				productMatrixThreeDerivative, SOFTMAX_SIZE, 0,
				leakyMatrixTwo, LEAKY_TWO_SIZE, 0,
				&GLOBAL::ONEF,
				parameters->weightMatrixThreeDerivative, SOFTMAX_SIZE, 0,
				1);
			cpuSgemmStridedBatched(
				true, false,
				LEAKY_TWO_SIZE, 1, SOFTMAX_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixThree, SOFTMAX_SIZE, 0,
				productMatrixThreeDerivative, SOFTMAX_SIZE, 0,
				&GLOBAL::ONEF,
				leakyMatrixTwoDerivative, LEAKY_TWO_SIZE, 0,
				1);
			cpuLeakyReluDerivative(leakyMatrixTwo, leakyMatrixTwoDerivative, productMatrixTwoDerivative, LEAKY_TWO_SIZE);
			cpuSgemmStridedBatched(
				false, true,
				LEAKY_TWO_SIZE, LEAKY_ONE_SIZE, 1,
				&GLOBAL::ONEF,
				productMatrixTwoDerivative, LEAKY_TWO_SIZE, 0,
				leakyMatrixOne, LEAKY_ONE_SIZE, 0,
				&GLOBAL::ONEF,
				parameters->weightMatrixTwoDerivative, LEAKY_TWO_SIZE, 0,
				1);
			cpuSgemmStridedBatched(
				true, false,
				LEAKY_ONE_SIZE, 1, LEAKY_TWO_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixTwo, LEAKY_TWO_SIZE, 0,
				productMatrixTwoDerivative, LEAKY_TWO_SIZE, 0,
				&GLOBAL::ONEF,
				leakyMatrixOneDerivative, LEAKY_ONE_SIZE, 0,
				1);
			cpuLeakyReluDerivative(leakyMatrixOne, leakyMatrixOneDerivative, productMatrixOneDerivative, LEAKY_ONE_SIZE);
			cpuSaxpy(LEAKY_ONE_SIZE, &GLOBAL::ONEF, productMatrixOneDerivative, 1, parameters->biasMatrixOneDerivative, 1);
			cpuSgemmStridedBatched(
				false, true,
				LEAKY_ONE_SIZE, INPUT_SIZE, 1,
				&GLOBAL::ONEF,
				productMatrixOneDerivative, LEAKY_ONE_SIZE, 0,
				inputMatrix, INPUT_SIZE, 0,
				&GLOBAL::ONEF,
				parameters->weightMatrixOneDerivative, LEAKY_ONE_SIZE, 0,
				1);
		}

		void Print()
		{
			/*PrintMatrix(inputMatrix, 1, INPUT_SIZE, "Input Matrix");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One + Bias");
			PrintMatrix(leakyMatrixOne, 1, LEAKY_ONE_SIZE, "Leaky Matrix One");
			PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two");
			PrintMatrix(leakyMatrixTwo, 1, LEAKY_TWO_SIZE, "Leaky Matrix Two");
			PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three");
			PrintMatrix(softmaxMatrix, 1, SOFTMAX_SIZE, "Softmax Matrix");
			printf("Sampled Action: %d\n", sampledAction);
			printf("Is Winner: %d\n", *isWinner);*/
			/*PrintMatrix(parameters->weightMatrixOneDerivative, LEAKY_ONE_SIZE, INPUT_SIZE, "Weight Matrix One Derivative");
			PrintMatrix(parameters->biasMatrixOneDerivative, 1, LEAKY_ONE_SIZE, "Bias Matrix One Derivative");
			PrintMatrix(parameters->weightMatrixTwoDerivative, LEAKY_TWO_SIZE, LEAKY_ONE_SIZE, "Weight Matrix Two Derivative");
			PrintMatrix(parameters->weightMatrixThreeDerivative, SOFTMAX_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Three Derivative");*/
			PrintMatrix(inputMatrix, 1, INPUT_SIZE, "Input Matrix");
			PrintMatrix(parameters->weightMatrixOne, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One");
			PrintMatrix(parameters->biasMatrixOne, 1, LEAKY_ONE_SIZE, "Bias Matrix One");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One");
		}
	};

	Parameters parameters;
	std::vector<Computation*> computations;

	uint32_t ForwardPropagate(float* board, float turn, bool* isWinner)
	{
		Computation* computation = new Computation();
		computations.emplace_back(computation);
		return computation->ForwardPropagate(&parameters, board, turn, isWinner);
	}

	void BackPropagate(float learningRate)
	{
		computations.back()->Print();
		for (auto computation : computations)
		{
			computation->BackPropagate();
			delete computation;
		}
		computations.clear();
		parameters.Update(learningRate * InvSqrt(computations.size()));
	}
};