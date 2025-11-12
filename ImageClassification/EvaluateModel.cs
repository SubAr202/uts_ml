using Microsoft.ML;
using Microsoft.ML.Data;
using System;

class EvaluateModel
{
    public static void Run()
    {
        var mlContext = new MLContext();

        // Lokasi model & data uji
        string modelPath = "model.zip";
        string testDataPath = "assets/inputs/images/tags.tsv";

        // Load model & data
        ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
        IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(testDataPath, hasHeader: false);

        // Evaluasi model
        var predictions = trainedModel.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:F2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");
    }
}

public class ImageData
{
    [LoadColumn(0)] public string ImagePath { get; set; }
    [LoadColumn(1)] public string Label { get; set; }
}
