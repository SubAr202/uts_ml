using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace ImageClassification.Train
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("🧠 === IMAGE CLASSIFICATION TRAINING ===");
            Console.ResetColor();

            // === 1. Path konfigurasi ===
            string datasetPath = args.Length > 0
                ? args[0]
                : @"D:\cloning\DeepLearning_ImageClassification_TensorFlow\ImageClassification\assets\inputs\images";

            string tagsTsv = Path.Combine(datasetPath, "tags.tsv");
            string outputModelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");
            string workspacePath = @"D:\model";

            Console.WriteLine("\n📁 === KONFIGURASI PATH ===");
            Console.WriteLine($"🔹 Dataset path : {datasetPath}");
            Console.WriteLine($"🔹 Label file   : {tagsTsv}");
            Console.WriteLine($"🔹 Model output : {outputModelPath}");
            Console.WriteLine();

            // === 2. Validasi file ===
            if (!Directory.Exists(datasetPath))
            {
                Error($"Folder dataset tidak ditemukan di: {datasetPath}");
                return;
            }

            if (!File.Exists(tagsTsv))
            {
                Error($"File tags.tsv tidak ditemukan di: {tagsTsv}");
                return;
            }

            // === 3. Inisialisasi MLContext ===
            var mlContext = new MLContext(seed: 1);

            Console.WriteLine("📥 Memuat data dari tags.tsv...");
            var data = mlContext.Data.LoadFromTextFile<ImageData>(
                path: tagsTsv,
                hasHeader: false,
                separatorChar: '\t');
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // === 4. Pipeline transformasi ===
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey",
                    inputColumnName: nameof(ImageData.Label))
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: datasetPath,
                    inputColumnName: nameof(ImageData.ImagePath)));

            var preprocessedTrainData = preprocessingPipeline.Fit(split.TrainSet).Transform(split.TrainSet);
            var preprocessedTestData = preprocessingPipeline.Fit(split.TrainSet).Transform(split.TestSet);

            // === 5. Konfigurasi trainer ===
            var options = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelKey",
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                ValidationSet = preprocessedTestData,
                Epoch = 20,
                BatchSize = 8,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) =>
                {
                    Console.WriteLine("📈 Training step selesai...");
                },
                TestOnTrainSet = false,
                WorkspacePath = workspacePath
            };

            // === 6. Training pipeline ===
            var trainingPipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(options)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n🚀 Training model sedang berlangsung... (harap tunggu)");
            Console.ResetColor();

            var trainedModel = trainingPipeline.Fit(preprocessedTrainData);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n✅ Training selesai!");
            Console.ResetColor();

            // === 7. Evaluasi ===
            Console.WriteLine("\n📊 Mengevaluasi model...");
            var predictions = trainedModel.Transform(preprocessedTestData);
            var metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelKey",
                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"\n📈 HASIL EVALUASI MODEL:");
            Console.WriteLine($"   🔹 Akurasi Mikro : {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"   🔹 Akurasi Makro : {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"   🔹 LogLoss       : {metrics.LogLoss:#.##}");

            // === 8. Simpan model ===
            mlContext.Model.Save(trainedModel, preprocessedTrainData.Schema, outputModelPath);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n💾 Model berhasil disimpan ke: {outputModelPath}");
            Console.ResetColor();
        }

        private static void Error(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"ERROR: {message}");
            Console.ResetColor();
        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath { get; set; }

            [LoadColumn(1)]
            public string Label { get; set; }
        }
    }
}
