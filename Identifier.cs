using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;
using System.Windows.Forms;

namespace FaceIdentifier
{
    public class Identifier
    {
        private List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        private List<string> labels = new List<string>();
        private HaarCascade faceDetected;

        public Identifier()
        {
            faceDetected = new HaarCascade("haarcascade_frontalface_default.xml");
        }

        public void Initialize(Dictionary<string, byte[]> imageDatabase)
        {
            foreach (var entry in imageDatabase)
            {
                using (MemoryStream ms = new MemoryStream(entry.Value))
                {
                    trainingImages.Add(new Image<Gray, byte>(new Bitmap(ms)));
                    labels.Add(entry.Key);
                }
            }
        }

        public byte[] CaptureFaceFromFile(string imagePath)
        {
            Image<Bgr, byte> uploadedImage = new Image<Bgr, byte>(imagePath);
            Image<Gray, byte> trainedFace = DetectAndCropFace(uploadedImage);

            if (trainedFace != null)
            {
                using (Bitmap bmp = trainedFace.ToBitmap())
                {
                    using (MemoryStream ms = new MemoryStream())
                    {
                        bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                        return ms.ToArray();
                    }
                }
            }
            else
            {
                throw new Exception("No face detected in the provided image.");
            }
        }

        public async Task<byte[]> CaptureFaceFromCamAsync(PictureBox pictureBox = null, int timeoutInSeconds = 10)
        {
            DateTime start = DateTime.Now;
            Image<Gray, byte> trainedFace = null;

            using (Emgu.CV.Capture camera = new Emgu.CV.Capture())
            {
                while ((DateTime.Now - start).TotalSeconds < timeoutInSeconds)
                {
                    Image<Bgr, byte> frame = camera.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                    if (pictureBox != null)
                    {
                        // Atualiza o PictureBox na thread de UI principal usando Invoke
                        pictureBox.Invoke((Action)(() =>
                        {
                            pictureBox.Image = frame.ToBitmap();
                            pictureBox.Refresh();
                        }));
                    }
                    trainedFace = DetectAndCropFace(frame);

                    if (trainedFace != null)
                    {
                        break;
                    }

                    await Task.Delay(100); // Wait a bit before trying again
                }
            }

            if (trainedFace != null)
            {
                using (Bitmap bmp = trainedFace.ToBitmap())
                {
                    using (MemoryStream ms = new MemoryStream())
                    {
                        bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                        return ms.ToArray();
                    }
                }
            }
            else
            {
                throw new Exception("No face detected within the timeout period.");
            }
        }

        public async Task<string> RecogAsync(PictureBox pictureBox = null, int timeoutInSeconds = 10)
        {
            DateTime start = DateTime.Now;
            string recognizedName = "-1";

            using (Emgu.CV.Capture camera = new Emgu.CV.Capture())
            {
                while ((DateTime.Now - start).TotalSeconds < timeoutInSeconds)
                {
                    Image<Bgr, byte> frame = camera.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                    Image<Gray, byte> grayFrame = frame.Convert<Gray, byte>();
                    if (pictureBox != null)
                    {
                        // Atualiza o PictureBox na thread de UI principal usando Invoke
                        pictureBox.Invoke((Action)(() =>
                        {
                            Console.WriteLine("Atualizando picturebox");
                            pictureBox.Image = frame.ToBitmap();
                            pictureBox.Refresh();
                        }));
                    }
                    MCvAvgComp[][] facesDetected = grayFrame.DetectHaarCascade(faceDetected, 1.2, 10, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20));

                    foreach (MCvAvgComp face in facesDetected[0])
                    {
                        Image<Gray, byte> result = frame.Copy(face.rect).Convert<Gray, byte>().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);

                        if (trainingImages.Count != 0)
                        {
                            MCvTermCriteria termCriterias = new MCvTermCriteria(trainingImages.Count, 0.001);
                            EigenObjectRecognizer recognizer = new EigenObjectRecognizer(trainingImages.ToArray(), labels.ToArray(), 1500, ref termCriterias);
                            recognizedName = recognizer.Recognize(result);

                            if (!string.IsNullOrEmpty(recognizedName))
                            {
                                return recognizedName;
                            }
                        }
                    }

                    await Task.Delay(100); // Wait a bit before trying again
                }
            }

            return recognizedName;
        }

        private Image<Gray, byte> DetectAndCropFace(Image<Bgr, byte> image)
        {
            Image<Gray, byte> grayImage = image.Convert<Gray, byte>();
            MCvAvgComp[][] facesDetected = grayImage.DetectHaarCascade(faceDetected, 1.2, 10, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20));

            foreach (MCvAvgComp face in facesDetected[0])
            {
                image.Draw(face.rect, new Bgr(Color.Green), 3);

                return grayImage.Copy(face.rect).Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            }

            return null;
        }
    }
}
