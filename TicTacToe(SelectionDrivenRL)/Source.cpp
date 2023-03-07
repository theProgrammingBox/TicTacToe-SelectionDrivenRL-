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

		Parameters()
		{
			cpuGenerateUniform(weightMatrixOne, WEIGHT_ONE_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(weightMatrixTwo, WEIGHT_TWO_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(weightMatrixThree, WEIGHT_THREE_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(biasMatrixOne, LEAKY_ONE_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(biasMatrixTwo, LEAKY_TWO_SIZE, -1.0f, 1.0f);
			cpuGenerateUniform(biasMatrixThree, SOFTMAX_SIZE, -1.0f, 1.0f);
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

		uint32_t SampleAction(Parameters* parameters, float* board, float turn, bool* isWinner)
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
			cpuSaxpy(LEAKY_TWO_SIZE, &GLOBAL::ONEF, parameters->biasMatrixTwo, 1, productMatrixTwo, 1);
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
			cpuSaxpy(SOFTMAX_SIZE, &GLOBAL::ONEF, parameters->biasMatrixThree, 1, productMatrixThree, 1);
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

		void Print()
		{
			/*PrintMatrix(inputMatrix, 1, INPUT_SIZE, "Input Matrix");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One");
			PrintMatrix(productMatrixOne, 1, LEAKY_ONE_SIZE, "Product Matrix One + Bias");
			PrintMatrix(leakyMatrixOne, 1, LEAKY_ONE_SIZE, "Leaky Matrix One");
			PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two");
			PrintMatrix(productMatrixTwo, 1, LEAKY_TWO_SIZE, "Product Matrix Two + Bias");
			PrintMatrix(leakyMatrixTwo, 1, LEAKY_TWO_SIZE, "Leaky Matrix Two");
			PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three");
			PrintMatrix(productMatrixThree, 1, SOFTMAX_SIZE, "Product Matrix Three + Bias");
			PrintMatrix(softmaxMatrix, 1, SOFTMAX_SIZE, "Softmax Matrix");*/
			printf("Sampled Action: %d\n", sampledAction);
			printf("Is Winner: %d\n", *isWinner);
		}
	};

	Parameters parameters;
	std::vector<Computation*> computations;

	uint32_t ForwardPropagate(float* board, float turn, bool* isWinner)
	{
		Computation* computation = new Computation();
		computations.emplace_back(computation);
		return computation->SampleAction(&parameters, board, turn, isWinner);
	}

	void BackPropagate()
	{
		for (auto computation : computations)
		{
			computation->Print();
		}
		for (auto computation : computations)
		{
			delete computation;
		}
		computations.clear();
	}
};

int main()
{
	constexpr uint32_t BOARD_WIDTH = 3;
	constexpr uint32_t BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH;
	NeuralNetwork network;

	//for (;;)
	for (uint32_t i = 1; i--;)
	{
		bool* playerOneWins = new bool(false);
		bool* playerTwoWins = new bool(false);
		
		printf("\nNew Game of TicTacToe\n");
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
			printf("\n");
			
			uint32_t playerInput;
			if (turn == 1.0f)
			{
				playerInput = network.ForwardPropagate(board, turn, playerOneWins);
			}
			else
			{
				playerInput = network.ForwardPropagate(board, turn, playerTwoWins);
			}
			
			if (board[playerInput] != 0)
			{
				gameRunning = false;
				if (turn == 1)
				{
					printf("Player 2 Wins due to Invalid Input\n");
					*playerTwoWins = true;
				}
				else
				{
					printf("Player 1 Wins due to Invalid Input\n");
					*playerOneWins = true;
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
				*playerOneWins = false;
				*playerTwoWins = false;
				break;
			}
			else if (row[rowPos] == BOARD_WIDTH || col[colPos] == BOARD_WIDTH || diagonal == BOARD_WIDTH || antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 2 Wins\n");
				*playerTwoWins = true;
				break;
			}
			else if (-row[rowPos] == BOARD_WIDTH || -col[colPos] == BOARD_WIDTH || -diagonal == BOARD_WIDTH || -antiDiagonal == BOARD_WIDTH)
			{
				gameRunning = false;
				printf("Player 1 Wins\n");
				*playerOneWins = true;
				break;
			}

		}
		
		network.BackPropagate();
		delete playerOneWins;
		delete playerTwoWins;
	}

	return 0;
}