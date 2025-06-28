
import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Home, Info, History, Menu, X, BarChart, PieChart, Layers, PieChart, Layers } from "lucide-react";

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  const isActive = (path: string) => location.pathname === path;

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? "py-2 bg-background/80 backdrop-blur-lg shadow-sm"
          : "py-4 bg-transparent"
      }`}
    >
      <div className="container flex justify-between items-center">
        <Link to="/" className="flex items-center space-x-2">
          <div className="relative w-8 h-8 rounded-lg bg-primary flex items-center justify-center overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-tr from-primary to-accent opacity-80"></div>
            <div className="z-10 text-white font-bold text-lg">DD</div>
          </div>
          <span className="font-medium text-xl">DeepDefend</span>
        </Link>

        {/* Desktop Navigation */}
        <NavigationMenu className="hidden md:flex">
          <NavigationMenuList>
            <NavigationMenuItem>
              <Link to="/">
                <NavigationMenuLink
                  className={`${navigationMenuTriggerStyle()} ${
                    isActive("/") ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <Home className="mr-2 h-4 w-4" />
                  Home
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link to="/about">
                <NavigationMenuLink
                  className={`${navigationMenuTriggerStyle()} ${
                    isActive("/about") ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <Info className="mr-2 h-4 w-4" />
                  About
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link to="/history">
                <NavigationMenuLink
                  className={`${navigationMenuTriggerStyle()} ${
                    isActive("/history") ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <History className="mr-2 h-4 w-4" />
                  History
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <NavigationMenuTrigger
                className={`${
                  isActive("/dashboard") || isActive("/full-dashboard") || isActive("/advanced-dashboard")
                    ? "bg-accent/10 text-accent"
                    : ""
                }`}
              >
                <BarChart className="mr-2 h-4 w-4" />
                Dashboards
              </N>
              <NavigationMenuContent>
                <ul className="grid gap-3 p-4 w-[220px]">
                  <li>
                    <NavigationMenuTrigger
                      className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                          isActive("/dashboard") ? "bg-accent/10 text-accent" : ""
                        }`}
                      >
                        <div className="text-sm font-medium leading-none">Simple Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Basic analytics and statistics
                        </p>
                      </NavigationMenuLink>
                    </Link>
                  </li>
                  <li>
                    <Link to="/full-dashboard">
                      <NavigationMenuLink
                        className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                          isActive("/full-dashboard") ? "bg-accent/10 text-accent" : ""
                        }`}
                      >
                        <div className="text-sm font-medium leading-none">Full Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Detailed analytics and visualizations
                        </p>
                      </NavigationMenuLink>
                    </Link>
                  </li>
                  <li>
                    <Link to="/advanced-dashboard">
                      <NavigationMenuLink
                        className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                        isActive("/advanced-dashboard") || isActive("/full-dashboard") || isActive("/advanced-dashboard")
                          ? "bg-accent/10 text-accent"
                    : ""
                      }`}
                    >
                      <div className="text-sm font-medium leading-none">Advanced Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Advanced metrics and settings
                        </p>
                      </NavigationMenuLink>
                    </Link>
                  </li>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link to="/multimodal-analysis">
                <NavigationMenuLink
                  className={`${navigationMenuTriggerStyle()} ${
                    isActive("/multimodal-analysis") ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <Layers className="mr-2 h-4 w-4" />
                Multi-Modal Analysis
              </NavigationMenuTrigger>
              <NavigationMenuContent>
                <ul className="grid gap-3 p-4 w-[220px]">
                  <li>
                    <Link to="/dashboard">
                      <NavigationMenuLink
                        className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                          isActive("/dashboard") ? "bg-accent/10 text-accent" : ""
                        }`}
                      >
                        <div className="text-sm font-medium leading-none">Simple Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Basic analytics and statistics
                        </p>
                      </NavigationMenuLink>
                    </Link>
                  </li>
                  <li>
                    <Link to="/full-dashboard">
                      <NavigationMenuLink
                        className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                          isActive("/full-dashboard") ? "bg-accent/10 text-accent" : ""
                        }`}
                      >
                        <div className="text-sm font-medium leading-none">Full Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Detailed analytics and visualizations
                        </p>
                      </NavigationMenuLink>
                    </Link>
                  </li>
                  <li>
                    <Link to="/advanced-dashboard">
                      <NavigationMenuLink
                        className={`block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground ${
                          isActive("/advanced-dashboard") ? "bg-accent/10 text-accent" : ""
                        }`}
                      >
                        <div className="text-sm font-medium leading-none">Advanced Dashboard</div>
                        <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                          Advanced metrics and settings
                        </p>
                <li>
                  <Link
                    to="/full-dashboard"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/full-dashboard")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <PieChart className="mr-3 h-5 w-5" />
                    Full Dashboard
                  </Link>
                </li>
                <li>
                  <Link
                    to="/advanced-dashboard"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/advanced-dashboard")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <BarChart className="mr-3 h-5 w-5" />
                    Advanced Dashboard
                  </Link>
                </li>
                <li>
                  <Link
                    to="/multimodal-analysis"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/multimodal-analysis")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <Layers className="mr-3 h-5 w-5" />
                    Multi-Modal Analysis
                  </Link>
                </li>
                      </NavigationMenuLink>
                  Simple   </Link>
                  </li>
                </ul>
              </NavigationMenuContent>
            </NavigationMenuItem>
            <NavigationMenuItem>
              <Link to="/multimodal-analysis">
                <NavigationMenuLink
                  className={`${navigationMenuTriggerStyle()} ${
                    isActive("/multimodal-analysis") ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <Layers className="mr-2 h-4 w-4" />
                  Multi-Modal Analysis
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>

        {/* Mobile Menu Button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          {isMobileMenuOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </Button>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="fixed inset-0 top-16 z-40 bg-background/95 backdrop-blur-sm md:hidden animate-fade-in">
            <nav className="container py-8">
              <ul className="flex flex-col space-y-4">
                <li>
                  <Link
                    to="/"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <Home className="mr-3 h-5 w-5" />
                    Home
                  </Link>
                </li>
                <li>
                  <Link
                    to="/about"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/about")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <Info className="mr-3 h-5 w-5" />
                    About
                  </Link>
                </li>
                <li>
                  <Link
                    to="/history"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/history")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <History className="mr-3 h-5 w-5" />
                    History
                  </Link>
                </li>
                <li>
                  <Link
                    to="/dashboard"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/dashboard")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <BarChart className="mr-3 h-5 w-5" />
                    Simple Dashboard
                  </Link>
                </li>
                <li>
                  <Link
                    to="/full-dashboard"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/full-dashboard")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <PieChart className="mr-3 h-5 w-5" />
                    Full Dashboard
                  </Link>
                </li>
                <li>
                  <Link
                    to="/advanced-dashboard"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/advanced-dashboard")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <BarChart className="mr-3 h-5 w-5" />
                    Advanced Dashboard
                  </Link>
                </li>
                <li>
                  <Link
                    to="/multimodal-analysis"
                    className={`flex items-center px-4 py-3 rounded-md ${
                      isActive("/multimodal-analysis")
                        ? "bg-accent/10 text-accent"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <Layers className="mr-3 h-5 w-5" />
                    Multi-Modal Analysis
                  </Link>
                </li>
              </ul>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
