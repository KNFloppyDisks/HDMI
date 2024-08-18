package linreg

import (
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
