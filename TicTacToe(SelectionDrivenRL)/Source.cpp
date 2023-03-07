#include "Header.h"

class NeuralNetwork
{
public:
	static constexpr uint32_t BOARD_WIDTH = 3;
	static constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;

	// 1x10 input
	// 10x20 weight
	// 1x20 product
	// 1x20 bias
	// 1x20 leaky activation
	// 20x30 weight
	// 1x30 product
	// 1x30 bias
	// 1x30 leaky activation
	// 30x9 weight
	// 1x9 product
	// 1x9 bias
	// 1x9 softmax activation
	
	static constexpr uint32_t INPUT_SIZE = BOARD_SIZE + 1;
	static constexpr uint32_t LEAKY_ONE_SIZE = 30;
	static constexpr uint32_t WEIGHT_ONE_SIZE = INPUT_SIZE * LEAKY_ONE_SIZE;
	static constexpr uint32_t LEAKY_TWO_SIZE = 20;
	static constexpr uint32_t WEIGHT_TWO_SIZE = LEAKY_ONE_SIZE * LEAKY_TWO_SIZE;
	static constexpr uint32_t SOFTMAX_SIZE = 9;
	static constexpr uint32_t WEIGHT_THREE_SIZE = LEAKY_TWO_SIZE * SOFTMAX_SIZE;

	struct Parameters
	{
		float weightMatrixOne[WEIGHT_ONE_SIZE];
		float biasMatrixOne[LEAKY_ONE_SIZE];
		float weightMatrixTwo[WEIGHT_TWO_SIZE];
		float biasMatrixTwo[LEAKY_TWO_SIZE];
		float weightMatrixThree[WEIGHT_THREE_SIZE];
		float biasMatrixThree[SOFTMAX_SIZE];
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
	};
	
	NeuralNetwork()
	{
		cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -1.0f, 1.0f);
		cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -1.0f, 1.0f);
		cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -1.0f, 1.0f);
		cpuGenerateUniform(biasMatrixOne, LEAKY_ONE_SIZE, -1.0f, 1.0f);
		cpuGenerateUniform(biasMatrixTwo, LEAKY_TWO_SIZE, -1.0f, 1.0f);
		cpuGenerateUniform(biasMatrixThree, SOFTMAX_SIZE, -1.0f, 1.0f);
	}

	~NeuralNetwork()
	{
	}

	uint32_t ForwardPropagate(float* board, float turn)
	{
		memcpy(inputMatrix, board, sizeof(int) * BOARD_SIZE);
		inputMatrix[BOARD_SIZE] = turn;
		//PrintMatrix(inputMatrix, 1, INPUT_SIZE, "Input Matrix");
		cpuSgemmStridedBatched(
			false, false,
			LEAKY_ONE_SIZE, 1, INPUT_SIZE,
			&GLOBAL::ONEF,
			weightMatrixOne, LEAKY_ONE_SIZE, 0,
			inputMatrix, INPUT_SIZE, 0,
			&GLOBAL::ZEROF,
			productMatrixOne, LEAKY_ONE_SIZE, 0,
			1);
		//PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One");
		cpuSaxpy(LEAKY_ONE_SIZE, &GLOBAL::ONEF, biasMatrixOne, 1, productMatrixOne, 1);
		//PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One + Bias");
		cpuLeakyRelu(productMatrixOne, leakyMatrixOne, LEAKY_ONE_SIZE);
		//PrintMatrix(leakyMatrixOne, 1, LEAKY_ONE_SIZE, "Leaky Matrix One");
		cpuSgemmStridedBatched(
			false, false,
			LEAKY_TWO_SIZE, 1, LEAKY_ONE_SIZE,
			&GLOBAL::ONEF,
			weightMatrixTwo, LEAKY_TWO_SIZE, 0,
			leakyMatrixOne, LEAKY_ONE_SIZE, 0,
			&GLOBAL::ZEROF,
			productMatrixTwo, LEAKY_TWO_SIZE, 0,
			1);
		//PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two");
		cpuSaxpy(LEAKY_TWO_SIZE, &GLOBAL::ONEF, biasMatrixTwo, 1, productMatrixTwo, 1);
		//PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two + Bias");
		cpuLeakyRelu(productMatrixTwo, leakyMatrixTwo, LEAKY_TWO_SIZE);
		//PrintMatrix(leakyMatrixTwo, 1, LEAKY_TWO_SIZE, "Leaky Matrix Two");
		cpuSgemmStridedBatched(
			false, false,
			SOFTMAX_SIZE, 1, LEAKY_TWO_SIZE,
			&GLOBAL::ONEF,
			weightMatrixThree, SOFTMAX_SIZE, 0,
			leakyMatrixTwo, LEAKY_TWO_SIZE, 0,
			&GLOBAL::ZEROF,
			productMatrixThree, SOFTMAX_SIZE, 0,
			1);
		//PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three");
		cpuSaxpy(SOFTMAX_SIZE, &GLOBAL::ONEF, biasMatrixThree, 1, productMatrixThree, 1);
		//PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three + Bias");
		cpuSoftmax(productMatrixThree, softmaxMatrix, SOFTMAX_SIZE);
		//PrintMatrix(softmaxMatrix, 1, SOFTMAX_SIZE, "Softmax Matrix");

		float randomNumber = GLOBAL::RANDOM.Rfloat();
		sampledAction = 0;
		for (;;)
		{
			randomNumber -= softmaxMatrix[sampledAction];
			if (randomNumber <= 0)
				break;
			sampledAction -= (++sampledAction >= SOFTMAX_SIZE) * SOFTMAX_SIZE;
		}
		printf("Sampled Action: %d\n", sampledAction);
		return sampledAction;
	}
};

int main()
{
	constexpr uint32_t BOARD_WIDTH = 3;
	constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;
	NeuralNetwork network;

	for (;;)
	{
		printf("\n\nNew Game of TicTacToe\n");
		uint32_t numMoves = 0;
		int row[BOARD_SIZE];
		int col[BOARD_SIZE];
		int diagonal = 0;
		int antiDiagonal = 0;
		float turn = 1;
		float board[BOARD_SIZE];
		memset(row, 0, sizeof(int) * BOARD_SIZE);
		memset(col, 0, sizeof(int) * BOARD_SIZE);
		memset(board, 0, sizeof(float) * BOARD_SIZE);

		bool gameRunning = true;
		while (gameRunning)
		{
			for (uint32_t i = 0; i < BOARD_WIDTH; i++)
			{
				for (uint32_t j = 0; j < BOARD_WIDTH; j++)
				{
					float value = board[i * BOARD_WIDTH + j];
					if (value == -1)
						printf("O");
					else if (value == 1)
						printf("X");
					else
						printf("-");
				}
				printf("\n");
			}
			
			network.ForwardPropagate(board, turn);
			uint32_t playerInput;
			printf("Enter position (0 - 8): ");
			scanf_s("%d", &playerInput);
			if (playerInput < 0 || playerInput > 8 || board[playerInput] != 0)
			{
				gameRunning = false;
				if (turn == 1)
				{
					printf("Player 2 Wins due to Invalid Input\n");
				}
				else
				{
					printf("Player 1 Wins due to Invalid Input\n");
				}
				break;
			}
			board[playerInput] = turn;
			*(int32_t*)&turn ^= 0x80000000;

			uint32_t rowPos = playerInput / BOARD_WIDTH;
			uint32_t colPos = playerInput % BOARD_WIDTH;
			row[rowPos] += turn;
			col[colPos] += turn;
			diagonal += turn * (rowPos == colPos);
			antiDiagonal += turn * (rowPos + colPos + 1 == BOARD_WIDTH);
			numMoves++;

			if (numMoves == BOARD_SIZE)
			{
				gameRunning = false;
				printf("Draw\n");
				break;
			}
			else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 2 Wins\n");
				break;
			}
			else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 1 Wins\n");
				break;
			}
		}
	}

	return 0;
}