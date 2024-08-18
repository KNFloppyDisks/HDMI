package linreg

import "gonum.org/v1/gonum/mat"

func MeanSquaredError(y_batch, P mat.Matrix) float64 {
	// Get the dimensions of the matrices
	r, c := y_batch.Dims()

	// Create a matrix to store the differences
	diff := mat.NewDense(r, c, nil)

	// Subtract P from y_batch and store the result in diff
	diff.Sub(y_batch, P)

	// Square each element in diff
	diff.Apply(func(i, j int, v float64) float64 {
		return v * v
	}, diff)

	// Calculate the sum of all elements in the diff matrix
	sum := mat.Sum(diff)

	// Calculate the mean squared error
	mse := sum / float64(r*c)

	return mse
}
