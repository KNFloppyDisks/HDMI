package linreg

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ErrorKind string

const (
	// either xBatchRows != yBatchRows or xBatchCols != wWeightsRows
	DimsDontMatch ErrorKind = "RowsDontMatch"

	// the weights["W"] does not exist
	NoWeights ErrorKind = "NoWeights"

	// the weights["B"] is not 1x1
	BWrongDimensions ErrorKind = "BWrongDimensions"
)

type Error struct {
	errorKind ErrorKind
	message   string
}

func (e Error) Error() string {
	return e.message
}

func NewError(ek ErrorKind, m string) Error {
	return Error{
		errorKind: ek,
		message:   m,
	}
}

type ForwardInfo struct {
	X mat.Matrix
	N mat.Matrix
	P mat.Matrix
	Y mat.Matrix
}

type LossForwardInfo struct {
	X  mat.Matrix
	M1 mat.Matrix
	N1 mat.Matrix
	O1 mat.Matrix
	M2 mat.Matrix
	P  mat.Matrix
	Y  mat.Matrix
}

func ForwardLinearRegression(xBatch, yBatch, weights mat.Matrix, bias float64) (float64, *ForwardInfo, error) {

	//First assert that all the dimensions and pieces are correctly supplied
	xBatchRows, xBatchCols := xBatch.Dims()
	yBatchRows, _ := yBatch.Dims()

	if xBatchRows != yBatchRows {
		return 0, nil, NewError(DimsDontMatch, "xBatchRows != yBatchRows")
	}

	weightsRows, weightsCols := weights.Dims()
	if xBatchCols != weightsRows {
		return 0, nil, NewError(DimsDontMatch, "xBatchCols != wWeightsRows")
	}

	// Now perform the dot product [mat.Dense]
	n := mat.NewDense(xBatchRows, weightsCols, nil)
	n.MulElem(xBatch, weights)

	p := mat.NewDense(xBatchRows, weightsCols, nil)

	// Add the b to the p
	p.Apply(func(i, j int, v float64) float64 {
		return v + bias
	}, p)

	// calculate loss
	// loss = np.mean(np.power(y_batch - P, 2))
	// Mean squared loss
	loss := MeanSquaredError(yBatch, p)

	forwardInfo := &ForwardInfo{
		X: xBatch,
		Y: yBatch,
		P: p,
		N: n,
	}

	return loss, forwardInfo, nil
}

func ForwardLoss(xBatch, yBatch, weights1, weights2 mat.Matrix, bias1, bias2 float64) (float64, *LossForwardInfo, error) {

	batchSize, numFeatures := xBatch.Dims()
	M1 := mat.NewDense(batchSize, numFeatures, nil)
	M1.MulElem(xBatch, weights1)

	// Sigmoid!
	N1 := mat.NewDense(batchSize, numFeatures, nil)
	N1.Apply(func(i, j int, v float64) float64 {
		return M1.At(i, j) + bias1
	}, N1)
	O1 := Sigmoid(N1)

	M2 := mat.NewDense(batchSize, numFeatures, nil)
	M2.MulElem(O1, weights2)
	P := mat.NewDense(batchSize, numFeatures, nil)
	P.Apply(func(i, j int, v float64) float64 {
		return M2.At(i, j) + bias2
	}, P)
	loss := MeanSquaredError(yBatch, P)

	lossForwardInfo := &LossForwardInfo{
		X:  xBatch,
		M1: M1,
		N1: N1,
		O1: O1,
		M2: M2,
		P:  P,
		Y:  yBatch,
	}
	return loss, lossForwardInfo, nil
}

// Sigmoid applies the sigmoid function element-wise to the input matrix.
func Sigmoid(matrix mat.Matrix) *mat.Dense {
	r, c := matrix.Dims()
	result := mat.NewDense(r, c, nil)

	// Apply sigmoid function element-wise
	result.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	}, matrix)

	return result
}
