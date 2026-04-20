#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS      1000
#define BATCH_SIZE 100

const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file  = "data/t10k-images-idx3-ubyte";
const char * test_labels_file  = "data/t10k-labels-idx1-ubyte";

/* Accuracy using full floating-point weights */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    for (i = 0, correct = 0; i < (int) dataset->size; i++) {
        neural_network_hypothesis(&dataset->images[i], network, activations);

        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    return ((float) correct) / ((float) dataset->size);
}

/* Accuracy using int8 quantized weights (PTQ or QAT-exported model) */
float calculate_accuracy_q(mnist_dataset_t * dataset, neural_network_q_t * network_q)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    for (i = 0, correct = 0; i < (int) dataset->size; i++) {
        neural_network_hypothesis_q(&dataset->images[i], network_q, activations);

        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[])
{
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t   batch;
    neural_network_t  network_fp;   /* floating-point / PTQ master weights */
    neural_network_t  network_qat;  /* QAT master weights (float, trained differently) */
    neural_network_q_t network_ptq_q;
    neural_network_q_t network_qat_q;
    float loss, accuracy_fp, accuracy_ptq, accuracy_qat;
    int   i, batches;

    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset  = mnist_get_dataset(test_images_file,  test_labels_file);
    batches = train_dataset->size / BATCH_SIZE;

    /* ------------------------------------------------------------------ */
    /* Phase 1: Standard floating-point training (also produces PTQ model) */
    /* ------------------------------------------------------------------ */
    printf("=== Phase 1: Floating-Point Training ===\n");
    neural_network_random_weights(&network_fp);

    for (i = 0; i < STEPS; i++) {
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);
        loss = neural_network_training_step(&batch, &network_fp, 0.5f);
        accuracy_fp = calculate_accuracy(test_dataset, &network_fp);
        printf("Step %04d\tLoss: %.4f\tFP Accuracy: %.3f\n",
               i, loss / batch.size, accuracy_fp);
    }

    accuracy_fp = calculate_accuracy(test_dataset, &network_fp);
    printf("\nFinal floating-point accuracy:  %.4f\n\n", accuracy_fp);

    /* ------------------------------------------------------------------ */
    /* Phase 2: Post-Training Quantization (PTQ) on the FP-trained model  */
    /* ------------------------------------------------------------------ */
    neural_network_post_training_quantize(&network_fp, &network_ptq_q);
    accuracy_ptq = calculate_accuracy_q(test_dataset, &network_ptq_q);
    printf("Final PTQ int8 accuracy:        %.4f\n\n", accuracy_ptq);

    if (neural_network_export_quantized_txt("quantized_ptq_model.txt", &network_ptq_q)) {
        printf("PTQ model saved to quantized_ptq_model.txt\n\n");
    }

    /* ------------------------------------------------------------------ */
    /* Phase 3: Quantization-Aware Training (QAT)                         */
    /*                                                                     */
    /* The master weights stay in float32 the entire time.  Every forward  */
    /* pass inside each training step fake-quantizes the weights (quantize */
    /* then dequantize) so the network sees the same rounding noise it     */
    /* will encounter at int8 inference.  Gradients are propagated back to */
    /* the float master weights via the straight-through estimator.        */
    /* ------------------------------------------------------------------ */
    printf("=== Phase 3: Quantization-Aware Training (QAT) ===\n");
    neural_network_random_weights(&network_qat);

    for (i = 0; i < STEPS; i++) {
        mnist_batch(train_dataset, &batch, BATCH_SIZE, i % batches);
        loss = neural_network_training_step_qat(&batch, &network_qat, 0.5f);
        accuracy_fp = calculate_accuracy(test_dataset, &network_qat);
        printf("Step %04d\tLoss: %.4f\tQAT (FP eval) Accuracy: %.3f\n",
               i, loss / batch.size, accuracy_fp);
    }

    /* Export QAT-trained float weights to int8 for final evaluation */
    neural_network_post_training_quantize(&network_qat, &network_qat_q);
    accuracy_qat = calculate_accuracy_q(test_dataset, &network_qat_q);
    printf("\nFinal QAT int8 accuracy:        %.4f\n\n", accuracy_qat);

    if (neural_network_export_quantized_txt("quantized_qat_model.txt", &network_qat_q)) {
        printf("QAT model saved to quantized_qat_model.txt\n");
    }

    /* ------------------------------------------------------------------ */
    /* Summary                                                             */
    /* ------------------------------------------------------------------ */
    printf("\n============================================================\n");
    printf("  Model Comparison Summary\n");
    printf("============================================================\n");
    printf("  Floating-Point (FP32):  %.4f  (%.2f%%)\n", accuracy_fp, accuracy_fp * 100.0f);
    printf("  Post-Training Quant:    %.4f  (%.2f%%)\n", accuracy_ptq, accuracy_ptq * 100.0f);
    printf("  Quant-Aware Training:   %.4f  (%.2f%%)\n", accuracy_qat, accuracy_qat * 100.0f);
    printf("============================================================\n");

    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
