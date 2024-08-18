package linreg

import (
	"gonum.org/v1/gonum/mat"
)

type LossGradientInfo struct {
	W mat.Matrix
	B float64
}

func LossGradient(info ForwardInfo, weights mat.Matrix, bias float64) *LossGradientInfo {
	batchSize, numOutputs := info.X.Dims()

	// dLdP = -2 * (forwardInfo['y'] - forwardInfo['P'])
	dLdP := mat.NewDense(batchSize, numOutputs, nil)
	dLdP.Sub(info.Y, info.P)
	dLdP.Scale(-2, dLdP)

	// dPdN = np.ones_like(forward_info['N'])
	dPdN := mat.NewDense(batchSize, numOutputs, nil)
	dPdN.Apply(func(i, j int, v float64) float64 { return 1.0 }, dPdN)

	// dPdB = np.ones_like(weights['B'])
	dPdB := mat.NewDense(batchSize, numOutputs, nil)
	dPdB.Apply(func(i, j int, v float64) float64 { return 1.0 }, dPdB)

	// dLdN = dLdP * dPdN
	dLdN := mat.NewDense(batchSize, numOutputs, nil)
	dLdN.MulElem(dLdP, dPdN)

	// dNdW = np.transpose(forward_info['X'], (1, 0))
	dNdW := mat.NewDense(numOutputs, batchSize, nil)
	dNdW.CloneFrom(info.X.T())

	// dLdW = np.dot(dNdW, dLdN)
	weightsRows, _ := weights.Dims()
	dLdW := mat.NewDense(weightsRows, numOutputs, nil)
	dLdW.Mul(dNdW, dLdN)

	// dLdB = (dLdP * dPdB).sum(axis=0)
	dLdB := 0.0
	for i := 0; i < batchSize; i++ {
		dLdB += dLdP.At(i, 0) * dPdB.At(i, 0)
	}

	lgi := &LossGradientInfo{
		W: dLdW,
		B: dLdB,
	}

	return lgi

}
