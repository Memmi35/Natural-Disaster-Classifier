"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { AlertTriangle, Shield, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  const router = useRouter()
  const [selectedRole, setSelectedRole] = useState<string | null>(null)

  const handleRoleSelect = (role: string) => {
    setSelectedRole(role)
    setTimeout(() => {
      router.push(`/${role.toLowerCase()}`)
    }, 500)
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-dark-bg">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <div className="flex items-center justify-center mb-4">
          <AlertTriangle className="h-12 w-12 text-accent-orange mr-2" />
          <h1 className="text-4xl font-bold text-white">Incident Analysis Dashboard</h1>
        </div>
        <p className="text-muted-foreground max-w-md mx-auto">
          AI-powered incident analysis for emergency responders and administrators
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className={`card-hover h-full ${selectedRole === "User" ? "border-accent-orange" : ""}`}>
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertTriangle className="h-6 w-6 text-accent-orange mr-2" />
                User Dashboard
              </CardTitle>
              <CardDescription>Upload and analyze incident photos</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Upload incident photos and receive AI-powered analysis including:
              </p>
              <ul className="text-sm text-muted-foreground space-y-2 mb-4">
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  Incident type and severity assessment
                </li>
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  Detailed scene analysis
                </li>
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  Recommended response actions
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button className="w-full group" onClick={() => handleRoleSelect("User")}>
                Continue as User
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </CardFooter>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className={`card-hover h-full ${selectedRole === "Admin" ? "border-accent-orange" : ""}`}>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Shield className="h-6 w-6 text-accent-orange mr-2" />
                Admin Dashboard
              </CardTitle>
              <CardDescription>Manage and review incident history</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">Access administrative tools and historical data:</p>
              <ul className="text-sm text-muted-foreground space-y-2 mb-4">
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  View all uploaded incidents
                </li>
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  Search and filter by various criteria
                </li>
                <li className="flex items-center">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent-orange mr-2"></div>
                  Access detailed analysis reports
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button className="w-full group" onClick={() => handleRoleSelect("Admin")} variant="outline">
                Continue as Admin
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
