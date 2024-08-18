package linreg

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type LossGradientInfo struct {
	W mat.Matrix
	B float64
}

func LossGradient(info ForwardInfo, weights mat.Matrix, bias float64) *LossGradientInfo {
	batchSize, numOutputs := info.X.Dims()

	fmt.Println("Dupa 1")
	// dLdP = -2 * (forwardInfo['y'] - forwardInfo['P'])
	dLdP := mat.NewDense(batchSize, numOutputs, nil)
	fmt.Println("Dupa 1.1")
	dLdP.Sub(info.Y, info.P)
	fmt.Println("Dupa 1.2")
	dLdP.Scale(-2, dLdP)

	fmt.Println("Dupa 2")
	// dPdN = np.ones_like(forward_info['N'])
	dPdN := mat.NewDense(batchSize, numOutputs, nil)
	dPdN.Apply(func(i, j int, v float64) float64 { return 1.0 }, dPdN)

	fmt.Println("Dupa 3")
	// dPdB = np.ones_like(weights['B'])
	dPdB := mat.NewDense(batchSize, numOutputs, nil)
	dPdB.Apply(func(i, j int, v float64) float64 { return 1.0 }, dPdB)

	fmt.Println("Dupa 4")
	// dLdN = dLdP * dPdN
	dLdN := mat.NewDense(batchSize, numOutputs, nil)
	dLdN.MulElem(dLdP, dPdN)

	fmt.Println("Dupa 5")
	// dNdW = np.transpose(forward_info['X'], (1, 0))
	dNdW := mat.NewDense(numOutputs, batchSize, nil)
	dNdW.CloneFrom(info.X.T())

	fmt.Println("Dupa 6")
	// dLdW = np.dot(dNdW, dLdN)
	weightsRows, _ := weights.Dims()
	dLdW := mat.NewDense(weightsRows, numOutputs, nil)
	dLdW.Mul(dNdW, dLdN)

	fmt.Println("Dupa 7")
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
