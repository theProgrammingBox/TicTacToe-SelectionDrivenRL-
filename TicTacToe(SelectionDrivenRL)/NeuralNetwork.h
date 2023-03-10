#pragma once
#include "Header.h"

class NeuralNetwork
{
public:
	static constexpr uint32_t BOARD_WIDTH = 3;
	static constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;

	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE + 1;
	static constexpr uint32_t LEAKY_ONE_SIZE = 30;
	static constexpr uint32_t WEIGHT_ONE_SIZE = INPUT_SIZE * LEAKY_ONE_SIZE;
	static constexpr uint32_t LEAKY_TWO_SIZE = 20;
	static constexpr uint32_t WEIGHT_TWO_SIZE = LEAKY_ONE_SIZE * LEAKY_TWO_SIZE;
	static constexpr uint32_t SOFTMAX_SIZE = 9;
	static constexpr uint32_t WEIGHT_THREE_SIZE = LEAKY_TWO_SIZE * SOFTMAX_SIZE;

	struct Parameters
	{
		/*float weightMatrixOne[WEIGHT_ONE_SIZE];
		float biasMatrixOne[LEAKY_ONE_SIZE];
		float weightMatrixTwo[WEIGHT_TWO_SIZE];
		float weightMatrixThree[WEIGHT_THREE_SIZE];

		float weightMatrixThreeDerivative[WEIGHT_THREE_SIZE];
		float weightMatrixTwoDerivative[WEIGHT_TWO_SIZE];
		float biasMatrixOneDerivative[LEAKY_ONE_SIZE];
		float weightMatrixOneDerivative[WEIGHT_ONE_SIZE];*/
		
		float* weightMatrixOne;
		float* biasMatrixOne;
		float* weightMatrixTwo;
		float* weightMatrixThree;

		float* weightMatrixThreeDerivative;
		float* weightMatrixTwoDerivative;
		float* biasMatrixOneDerivative;
		float* weightMatrixOneDerivative;

		Parameters()
		{
			weightMatrixOne = new float[WEIGHT_ONE_SIZE];
			biasMatrixOne = new float[LEAKY_ONE_SIZE];
			weightMatrixTwo = new float[WEIGHT_TWO_SIZE];
			weightMatrixThree = new float[WEIGHT_THREE_SIZE];
			
			weightMatrixThreeDerivative = new float[WEIGHT_THREE_SIZE];
			weightMatrixTwoDerivative = new float[WEIGHT_TWO_SIZE];
			biasMatrixOneDerivative = new float[LEAKY_ONE_SIZE];
			weightMatrixOneDerivative = new float[WEIGHT_ONE_SIZE];
			
			cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)), sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)));
			cpuGenerateUniform(biasMatrixOne, LEAKY_ONE_SIZE, -sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)), sqrt(6.0f / (INPUT_SIZE + LEAKY_ONE_SIZE)));
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -sqrt(6.0f / (LEAKY_ONE_SIZE + LEAKY_TWO_SIZE)), sqrt(6.0f / (LEAKY_ONE_SIZE + LEAKY_TWO_SIZE)));
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -sqrt(6.0f / (LEAKY_TWO_SIZE + SOFTMAX_SIZE)), sqrt(6.0f / (LEAKY_TWO_SIZE + SOFTMAX_SIZE)));
			
			/*cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -0.01f, 0.01f);
			cpuGenerateUniform(biasMatrixOne, LEAKY_ONE_SIZE, -0.01f, 0.01f);
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -0.01f, 0.01f);
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -0.01f, 0.01f);*/
			
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
			
			/*PrintMatrix(weightMatrixThree, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three");
			PrintMatrix(weightMatrixTwo, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two");
			PrintMatrix(biasMatrixOne, 1, LEAKY_ONE_SIZE, "Bias Matrix One");
			PrintMatrix(weightMatrixOne, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One");*/

			/*cpuSaxpy(WEIGHT_THREE_SIZE, &learningRate, weightMatrixThreeDerivative, 1, weightMatrixThree, 1);
			cpuSaxpy(WEIGHT_TWO_SIZE, &learningRate, weightMatrixTwoDerivative, 1, weightMatrixTwo, 1);
			cpuSaxpy(LEAKY_ONE_SIZE, &learningRate, biasMatrixOneDerivative, 1, biasMatrixOne, 1);
			cpuSaxpy(WEIGHT_ONE_SIZE, &learningRate, weightMatrixOneDerivative, 1, weightMatrixOne, 1);*/

			/*PrintMatrix(weightMatrixThreeDerivative, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three Derivative");
			PrintMatrix(weightMatrixTwoDerivative, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two Derivative");
			PrintMatrix(biasMatrixOneDerivative, 1, LEAKY_ONE_SIZE, "Bias Matrix One Derivative");
			PrintMatrix(weightMatrixOneDerivative, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One Derivative");*/

			/*PrintMatrix(weightMatrixThree, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three");
			PrintMatrix(weightMatrixTwo, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two");
			PrintMatrix(biasMatrixOne, 1, LEAKY_ONE_SIZE, "Bias Matrix One");
			PrintMatrix(weightMatrixOne, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One");*/
			
			memset(weightMatrixThreeDerivative, 0, sizeof(float) * WEIGHT_THREE_SIZE);
			memset(weightMatrixTwoDerivative, 0, sizeof(float) * WEIGHT_TWO_SIZE);
			memset(biasMatrixOneDerivative, 0, sizeof(float) * LEAKY_ONE_SIZE);
			memset(weightMatrixOneDerivative, 0, sizeof(float) * WEIGHT_ONE_SIZE);
		}

		void Print()
		{
			PrintMatrix(weightMatrixOneDerivative, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One Derivative");
			PrintMatrix(weightMatrixOne, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One");
			PrintMatrix(biasMatrixOneDerivative, 1, LEAKY_ONE_SIZE, "Bias Matrix One Derivative");
			PrintMatrix(biasMatrixOne, 1, LEAKY_ONE_SIZE, "Bias Matrix One");
			PrintMatrix(weightMatrixTwoDerivative, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two Derivative");
			PrintMatrix(weightMatrixTwo, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two");
			PrintMatrix(weightMatrixThreeDerivative, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three Derivative");
			PrintMatrix(weightMatrixThree, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three");
		}
	};

	struct Computation
	{
		Parameters* parameters;
		/*float inputMatrix[INPUT_SIZE];
		float productMatrixOne[LEAKY_ONE_SIZE];
		float leakyMatrixOne[LEAKY_ONE_SIZE];
		float productMatrixTwo[LEAKY_TWO_SIZE];
		float leakyMatrixTwo[LEAKY_TWO_SIZE];
		float productMatrixThree[SOFTMAX_SIZE];
		float softmaxMatrix[SOFTMAX_SIZE];*/

		float* inputMatrix;
		float* productMatrixOne;
		float* leakyMatrixOne;
		float* productMatrixTwo;
		float* leakyMatrixTwo;
		float* productMatrixThree;
		float* softmaxMatrix;
		
		uint32_t sampledAction;
		bool* isWinner;

		Computation()
		{
			inputMatrix = (float*)malloc(sizeof(float) * INPUT_SIZE);
			productMatrixOne = (float*)malloc(sizeof(float) * LEAKY_ONE_SIZE);
			leakyMatrixOne = (float*)malloc(sizeof(float) * LEAKY_ONE_SIZE);
			productMatrixTwo = (float*)malloc(sizeof(float) * LEAKY_TWO_SIZE);
			leakyMatrixTwo = (float*)malloc(sizeof(float) * LEAKY_TWO_SIZE);
			productMatrixThree = (float*)malloc(sizeof(float) * SOFTMAX_SIZE);
			softmaxMatrix = (float*)malloc(sizeof(float) * SOFTMAX_SIZE);
		}

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
			PrintMatrix(leakyMatrixOne, 1, LEAKY_ONE_SIZE, "Leaky Matrix One");
			cpuSgemmStridedBatched(
				false, false,
				LEAKY_TWO_SIZE, 1, LEAKY_ONE_SIZE,
				&GLOBAL::ONEF,
				parameters->weightMatrixTwo, LEAKY_TWO_SIZE, 0,
				leakyMatrixOne, LEAKY_ONE_SIZE, 0,
				&GLOBAL::ZEROF,
				productMatrixTwo, LEAKY_TWO_SIZE, 0,
				1);
			PrintMatrix(parameters->weightMatrixTwo, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two");
			PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two");
			cpuLeakyRelu(productMatrixTwo, leakyMatrixTwo, LEAKY_TWO_SIZE);
			PrintMatrix(leakyMatrixTwo, 1, LEAKY_TWO_SIZE, "Leaky Matrix Two");
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
			PrintMatrix(softmaxMatrix, 1, SOFTMAX_SIZE, "Softmax Matrix");

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
			
			cpuSoftmaxDerivative(softmaxMatrix, productMatrixThreeDerivative, *isWinner, sampledAction, SOFTMAX_SIZE);
			//PrintMatrix(productMatrixThreeDerivative, SOFTMAX_SIZE, 1, "productMatrixThreeDerivative");
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
			//PrintMatrix(leakyMatrixTwoDerivative, LEAKY_TWO_SIZE, 1, "leakyMatrixTwoDerivative");
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
			//PrintMatrix(leakyMatrixOneDerivative, LEAKY_ONE_SIZE, 1, "leakyMatrixOneDerivative");
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
			PrintMatrix(inputMatrix, 1, INPUT_SIZE, "Input Matrix");
			PrintMatrix(parameters->weightMatrixOne, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One");
			PrintMatrix(parameters->weightMatrixOneDerivative, INPUT_SIZE, LEAKY_ONE_SIZE, "Weight Matrix One Derivative");
			PrintMatrix(parameters->biasMatrixOne, 1, LEAKY_ONE_SIZE, "Bias Matrix One");
			PrintMatrix(parameters->biasMatrixOneDerivative, 1, LEAKY_ONE_SIZE, "Bias Matrix One Derivative");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One");
			PrintMatrix(leakyMatrixOne, 1, LEAKY_ONE_SIZE, "Leaky Matrix One");
			PrintMatrix(parameters->weightMatrixTwo, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two");
			PrintMatrix(parameters->weightMatrixTwoDerivative, LEAKY_ONE_SIZE, LEAKY_TWO_SIZE, "Weight Matrix Two Derivative");
			PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two");
			PrintMatrix(leakyMatrixTwo, 1, LEAKY_TWO_SIZE, "Leaky Matrix Two");
			PrintMatrix(parameters->weightMatrixThree, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three");
			PrintMatrix(parameters->weightMatrixThreeDerivative, LEAKY_TWO_SIZE, SOFTMAX_SIZE, "Weight Matrix Three Derivative");
			PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three");
			PrintMatrix(softmaxMatrix, 1, SOFTMAX_SIZE, "Softmax Matrix");
			printf("Sampled Action: %d\n", sampledAction);
			printf("Is Winner: %d\n", *isWinner);
		}
	};

	Parameters parameters;
	std::vector<Computation*> computations;

	~NeuralNetwork()
	{
		delete[] parameters.weightMatrixOne;
		delete[] parameters.biasMatrixOne;
		delete[] parameters.weightMatrixTwo;
		delete[] parameters.weightMatrixThree;
			
		delete[] parameters.weightMatrixOneDerivative;
		delete[] parameters.biasMatrixOneDerivative;
		delete[] parameters.weightMatrixTwoDerivative;
		delete[] parameters.weightMatrixThreeDerivative;
	}

	uint32_t ForwardPropagate(float* board, float turn, bool* isWinner)
	{
		Computation* computation = new Computation();
		computations.emplace_back(computation);
		return computation->ForwardPropagate(&parameters, board, turn, isWinner);
	}

	void BackPropagate(float learningRate)
	{
		for (auto computation : computations)
		{
			if (*computation->isWinner == false)
			{
				computation->BackPropagate();
			}
		}
		//computations.back()->Print();
		parameters.Update(learningRate * InvSqrt(computations.size()));
		for (auto computation : computations)
		{
			delete[] computation->inputMatrix;
			delete[] computation->productMatrixOne;
			delete[] computation->leakyMatrixOne;
			delete[] computation->productMatrixTwo;
			delete[] computation->leakyMatrixTwo;
			delete[] computation->productMatrixThree;
			delete[] computation->softmaxMatrix;
			delete computation;
		}
		computations.clear();
	}

	void Print()
	{
		parameters.Print();
	}
};