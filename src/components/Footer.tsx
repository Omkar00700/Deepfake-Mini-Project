
import { Github } from "lucide-react";

const Footer = () => {
  // Get current year dynamically
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="border-t py-6 md:py-0">
      <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
        <div className="flex flex-col items-center gap-4 px-8 md:flex-row md:gap-2 md:px-0">
          <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
            &copy; {currentYear} DeepDefend AI. All rights reserved.
          </p>
        </div>
        <div className="flex items-center">
          <a 
            href="https://github.com/deepdefend" 
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground flex items-center"
          >
            <Github className="h-4 w-4 mr-1" />
            <span className="text-sm">GitHub</span>
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
