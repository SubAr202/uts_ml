using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.Predict
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("🔍 === IMAGE CLASSIFICATION PREDIKSI ===");
            Console.ResetColor();

            // === 1. Cari file model ===
            string[] possiblePaths =
            {
                Path.Combine(Environment.CurrentDirectory, "model.zip"),
                Path.Combine(@"D:\cloning\DeepLearning_ImageClassification_TensorFlow\", "model.zip"),
                Path.Combine(@"D:\cloning\DeepLearning_ImageClassification_TensorFlow\ImageClassification.Train", "model.zip"),
                Path.Combine(@"D:\cloning\DeepLearning_ImageClassification_TensorFlow\ImageClassification.Train\bin\Debug\net8.0", "model.zip")
            };

            string modelPath = possiblePaths.FirstOrDefault(File.Exists)
                               ?? throw new FileNotFoundException("❌ Tidak dapat menemukan file model.zip di lokasi yang diketahui.");

            Console.WriteLine($"📦 Menggunakan model: {modelPath}");

            // === 2. Path gambar untuk prediksi ===
            string imageRoot = @"D:\cloning\DeepLearning_ImageClassification_TensorFlow\ImageClassification\assets\inputs\images";
            string folder = Path.Combine(imageRoot, "plastic"); 

            if (!Directory.Exists(folder))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ Folder gambar tidak ditemukan: {folder}");
                Console.ResetColor();
                return;
            }

            var images = Directory.GetFiles(folder, "*.jpg", SearchOption.TopDirectoryOnly);
            if (images.Length == 0)
            {
                Console.WriteLine($"❌ Tidak ada file JPG di folder: {folder}");
                return;
            }

            // Ambil gambar pertama jika tidak ada argumen
            string imagePath = args.Length > 0 ? args[0] : images[0];
            if (!File.Exists(imagePath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ Gambar tidak ditemukan: {imagePath}");
                Console.ResetColor();
                return;
            }

            Console.WriteLine($"\n🖼️  Menguji gambar: {Path.GetFileName(imagePath)}");

            // === 3. Load model ===
            var mlContext = new MLContext();
            DataViewSchema modelSchema;
            var trainedModel = mlContext.Model.Load(modelPath, out modelSchema);

            // === 4. Buat pipeline kecil agar sesuai schema training ===
            var imageData = new[] { new ImageInput { ImagePath = imagePath } };
            var imageDataView = mlContext.Data.LoadFromEnumerable(imageData);
            var pipeline = mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: Path.GetDirectoryName(imagePath),
                inputColumnName: nameof(ImageInput.ImagePath));

            var imageDataTransformed = pipeline.Fit(imageDataView).Transform(imageDataView);

            // === 5. Gunakan model untuk prediksi ===
            var predictions = trainedModel.Transform(imageDataTransformed);

            // === 6. Ambil hasil prediksi ===
            var prediction = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false).First();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✅ Prediksi selesai!");
            Console.WriteLine($"   🔹 Gambar           : {Path.GetFileName(imagePath)}");
            Console.WriteLine($"   🔹 Label Terprediksi: {prediction.PredictedLabel}");
            Console.WriteLine($"   🔹 Skor Probabilitas: {prediction.Score.Max():P2}");
            Console.ResetColor();
        }

        // === Class input & output ===
        public class ImageInput
        {
            public string ImagePath { get; set; } = string.Empty;
        }

        public class ImagePrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; } = string.Empty;

            [ColumnName("Score")]
            public float[] Score { get; set; } = Array.Empty<float>();
        }
    }
}
