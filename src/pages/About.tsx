
import { motion } from "framer-motion";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const About = () => {
  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      
      <main className="flex-1 pt-24 pb-16">
        <div className="container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl mx-auto"
          >
            <div className="text-center mb-12">
              <h1 className="text-4xl font-bold mb-4">About DeepDefend</h1>
              <p className="text-lg text-muted-foreground">
                Our mission and technology
              </p>
            </div>
            
            <Card className="glass-panel mb-10">
              <CardContent className="pt-6">
                <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
                <p className="text-muted-foreground leading-relaxed mb-6">
                  DeepDefend was created to address the growing concern of deepfake technology targeting Indian identities. 
                  As deepfake technology becomes more sophisticated and accessible, the need for effective detection tools 
                  becomes increasingly critical, especially for regions with unique facial characteristics that general models 
                  might not accurately analyze.
                </p>
                <p className="text-muted-foreground leading-relaxed">
                  We aim to protect the authenticity of digital content, preserve trust in visual media, and empower 
                  individuals and organizations to verify the legitimacy of the images they encounter. By focusing specifically 
                  on Indian faces, we provide a specialized tool that fills a crucial gap in existing detection technologies.
                </p>
              </CardContent>
            </Card>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <h2 className="text-2xl font-semibold mb-6">Our Technology</h2>
              
              <div className="space-y-8">
                <div>
                  <h3 className="text-xl font-medium mb-3">Model Architecture</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Our deepfake detection system utilizes a sophisticated convolutional neural network architecture, 
                    specifically optimized to identify the subtle artifacts and inconsistencies present in manipulated images.
                    The model has been trained on a diverse dataset of both authentic and deepfake images, with special 
                    emphasis on the unique features of Indian faces.
                  </p>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="text-xl font-medium mb-3">Training Data</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    The model was initially trained on the FaceForensics++ dataset, which contains a large collection of 
                    facial manipulation examples. It was then fine-tuned with the Indian Deepfake Corpus, a specialized 
                    dataset created specifically for improving detection accuracy for Indian faces. This two-stage training 
                    process ensures both general effectiveness and specialized performance.
                  </p>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="text-xl font-medium mb-3">Detection Process</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    When you upload an image, our system performs the following steps:
                  </p>
                  <ol className="list-decimal list-inside space-y-2 mt-4 text-muted-foreground">
                    <li>Face detection and extraction from the uploaded image</li>
                    <li>Preprocessing of the facial region for optimal analysis</li>
                    <li>Analysis by our deep learning model to identify manipulation patterns</li>
                    <li>Calculation of a probability score indicating likelihood of manipulation</li>
                    <li>Results display with visual indicators of authenticity</li>
                  </ol>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="text-xl font-medium mb-3">Accuracy and Limitations</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    While our system demonstrates high accuracy in detecting many types of deepfakes, it's important to note 
                    that deepfake technology is continually evolving. Our model is regularly updated to improve detection 
                    capabilities, but no system can guarantee 100% accuracy in all cases. We recommend using DeepDefend as 
                    one tool in a broader verification strategy for critical content assessment.
                  </p>
                </div>
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mt-12 p-6 bg-primary/5 rounded-lg border border-primary/10"
            >
              <h2 className="text-xl font-semibold mb-4">Privacy Commitment</h2>
              <p className="text-muted-foreground leading-relaxed">
                We take your privacy seriously. All uploaded images are processed securely and are not shared with third parties. 
                While detection results are logged for system improvement, the images themselves are not permanently stored on our servers. 
                For more information, please contact us directly.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default About;
