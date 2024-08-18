package main

import (
	"fmt"
	"hdmi/linreg"
	"log"

	"gonum.org/v1/gonum/mat"
)

func main() {
	x := mat.NewDense(3, 3, []float64{
		4, 3, 2,
		5, 3, 4,
		9, 10, 3,
	})

	y := mat.NewDense(3, 3, []float64{
		12, 4, 23,
		5, 9, 8,
		7, 14, 15,
	})

	w := mat.NewDense(3, 3, []float64{
		1, 2, 2,
		3, 2, 0,
		2, 1, 1,
	})

	b := 2.00

	loss, info, err := linreg.ForwardLinearRegression(x, y, w, b)
	if err != nil {
		log.Printf("in ForwardLinearRegression: %v", err)
	}
	fmt.Printf("LOSS: %f\n", loss)

	lgi := linreg.LossGradient(*info, w, b)

	fmt.Printf("lgi.1: %f\nlgi.2: %f\n", lgi.B, lgi.W)
}
