package main

import (
	"fmt"
	"hdmi/linreg"
	"log"

	"gonum.org/v1/gonum/mat"
)

func main() {
	x := mat.NewDense(3, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	y := mat.NewDense(3, 3, []float64{
		1, 4, 7,
		2, 5, 8,
		3, 6, 9,
	})

	w := mat.NewDense(3, 3, []float64{
		0.5, 1, 1.5,
		2, 2.5, 3,
		3.5, 4, 4.5,
	})

	w2 := w.T()

	b := 1.00

	loss, info, err := linreg.ForwardLinearRegression(x, y, w, b)
	if err != nil {
		log.Printf("in ForwardLinearRegression: %v", err)
	}
	fmt.Printf("LOSS: %f\n", loss)

	lgi := linreg.LossGradient(*info, w, b)

	fmt.Printf("lgi.Bias: %f\nlgi.Weights: %f\n", lgi.B, lgi.W)

	fl, _, err := linreg.ForwardLoss(x, y, w, w2, b, -1.00)

	fmt.Printf("fl LOSS: %f", fl)
}
