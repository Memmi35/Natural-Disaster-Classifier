"use client"

import type React from "react"
import { useState, useRef } from "react"
import { GoogleGenerativeAI } from "@google/generative-ai"
import {
  AlertTriangle,
  Upload,
  Clock,
  BarChart,
  ArrowLeft,
  Loader2,
  Cloud,
  Waves,
  Flame,
  CloudLightning,
  AlertCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import Link from "next/link"

// Initialize the Generative AI instance
// You would get this API key from your environment variables in production
const genAI = new GoogleGenerativeAI("AIzaSyDrAZyhNuy1QslkpcFNH8r1whHbWRpkYKs");

export default function UserDashboard() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    // Check if file is an image
    if (!file.type.match("image.*")) {
      alert("Please upload an image file (jpg or png)")
      return
    }

    setFile(file)
    setError(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    // Reset analysis
    setAnalysisResult(null)
  }

  const getImageData = async (file: File): Promise<Uint8Array> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer
        const uint8Array = new Uint8Array(arrayBuffer)
        resolve(uint8Array)
      }
      reader.onerror = reject
      reader.readAsArrayBuffer(file)
    })
  }

  const handleAnalyze = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setError(null)

    try {
      // Get binary data from the file
      const imageData = await getImageData(file)
      
      // Set up the model and prepare prompt
      const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" })
      
      // Create a prompt with specific instructions for disaster detection
      const prompt = `
        Analyze this image and determine if it shows any of these specific disasters: earthquake, fog, heavy_rain, or on_fire (for fire/burning scenes).
        
        Respond in JSON format with the following structure:
        {
          "predictions": {
            "earthquake": boolean,
            "fog": boolean,
            "heavy_rain": boolean,
            "on_fire": boolean
          },
          "primaryPrediction": "most likely disaster type or null if none detected",
          "confidence": number (90-100),
          "summary": "brief description of what's visible",
          "details": "more detailed analysis of the scene",
          "severity": "Low, Medium, or High based on visual assessment"
        }
      `

      // Create content parts
      const imagePart = {
        inlineData: {
          data: Buffer.from(imageData).toString("base64"),
          mimeType: file.type
        }
      }

      // Generate content using the model
      const result = await model.generateContent([prompt, imagePart])
      const response = await result.response
      const text = response.text()
      
      // Parse the JSON response
      let parsedResponse
      try {
        // Extract JSON from the text (it might be wrapped in markdown code blocks)
        const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/) || text.match(/```([\s\S]*?)```/) || [null, text]
        const jsonString = jsonMatch[1] || text
        parsedResponse = JSON.parse(jsonString.trim())
      } catch (e) {
        console.error("Error parsing JSON response:", e)
        throw new Error("Failed to parse API response")
      }

      // Generate recommendations based on the detected disaster
      const recommendations = generateRecommendations(parsedResponse.primaryPrediction)
      
      // Format the final analysis result
      const formattedResult = {
        summary: parsedResponse.summary || "Disaster scene analyzed.",
        details: parsedResponse.details || "The image has been analyzed for potential disaster scenarios.",
        severity: parsedResponse.severity || "Unknown",
        confidence: parsedResponse.confidence || 0,
        timestamp: new Date().toISOString(),
        tags: getTagsFromPredictions(parsedResponse.predictions),
        primaryDisaster: parsedResponse.primaryPrediction || null,
        recommendations: recommendations,
      }
      
      setAnalysisResult(formattedResult)
    } catch (err) {
      console.error("Error analyzing image:", err)
      setError("Failed to analyze image. Please try again or check your API key configuration.")
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  const getTagsFromPredictions = (predictions: Record<string, boolean>) => {
    if (!predictions) return ["Analysis failed"]
    
    return Object.entries(predictions)
      .filter(([_, isPresent]) => isPresent)
      .map(([disasterType]) => disasterType)
  }
  
  const generateRecommendations = (disasterType: string | null) => {
    const commonRecommendations = [
      "Ensure you are in a safe location",
      "Follow instructions from local authorities",
      "Check on vulnerable neighbors if safe to do so",
      "Monitor emergency broadcasts for updates"
    ]
    
    if (!disasterType) return commonRecommendations
    
    switch (disasterType) {
      case "earthquake":
        return [
          "Drop, cover, and hold on",
          "Stay away from windows and exterior walls",
          "If outdoors, stay in open areas away from buildings",
          "Be prepared for aftershocks",
          ...commonRecommendations
        ]
      case "fog":
        return [
          "Reduce speed and use low-beam headlights when driving",
          "Maintain extra distance between vehicles",
          "Avoid unnecessary travel until conditions improve",
          "Use fog lights if available",
          ...commonRecommendations
        ]
      case "heavy_rain":
        return [
          "Avoid driving through flooded areas",
          "Stay away from fast-moving water",
          "Move to higher ground if in a flood-prone area",
          "Prepare for possible power outages",
          ...commonRecommendations
        ]
      case "on_fire":
        return [
          "Evacuate immediately if in danger",
          "Cover nose and mouth with a wet cloth if smoke is present",
          "Stay low to the ground if smoke is in the area",
          "Do not return until authorities declare it safe",
          ...commonRecommendations
        ]
      default:
        return commonRecommendations
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setAnalysisResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getDisasterIcon = (disasterType: string) => {
    switch (disasterType) {
      case "earthquake":
        return <AlertTriangle className="h-5 w-5" />
      case "fog":
        return <Cloud className="h-5 w-5" />
      case "heavy_rain":
        return <Waves className="h-5 w-5" />
      case "on_fire":
        return <Flame className="h-5 w-5" />
      default:
        return <AlertCircle className="h-5 w-5" />
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center gap-4 mb-8">
        <h1 className="text-3xl font-bold">Disaster Scene Analysis</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload Area */}
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle>Upload Image</CardTitle>
            <CardDescription>
              Upload an image to analyze for disaster scenarios
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!preview ? (
              <div
                className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                  isDragging ? "border-primary bg-muted/50" : "border-muted-foreground/25"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="flex flex-col items-center gap-2">
                  <Upload className="h-10 w-10 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground mt-2">
                    Drag and drop an image here, or click to select a file
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileInput}
                  />
                </div>
              </div>
            ) : (
              <div className="relative">
                <img src={preview} alt="Preview" className="w-full rounded-lg object-contain max-h-64" />
                <Button
                  variant="outline"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={handleReset}
                >
                  Change
                </Button>
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-red-100 border border-red-300 rounded-md text-red-800 text-sm">
                <p className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" /> {error}
                </p>
              </div>
            )}
          </CardContent>
          <CardFooter className="flex justify-end">
            <Button
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing}
              className="flex items-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <BarChart className="h-4 w-4" />
                  Analyze Image
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        {/* Results Area */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              {analysisResult
                ? `Analysis completed at ${new Date(analysisResult.timestamp).toLocaleTimeString()}`
                : "Upload and analyze an image to see results"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {analysisResult ? (
              <div className="space-y-4">
                {/* <div className="flex flex-wrap gap-2 mb-2">
                  {analysisResult.tags.map((tag: string, i: number) => (
                    <Badge key={i} variant="outline" className="flex items-center gap-1">
                      {getDisasterIcon(tag)}
                      {tag.replace("_", " ")}
                    </Badge>
                  ))}
                </div> */}

                <div className="flex flex-wrap gap-3 mb-4 p-2 justify-center bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 rounded-lg shadow-lg">
                  {analysisResult.tags.map((tag, i) => (
                    <Badge
                      key={i}
                      variant="filled"
                      className="flex items-center gap-2 px-4 py-2 text-lg font-semibold rounded-full bg-gradient-to-r from-purple-600 via-pink-500 to-red-500 text-white shadow-md hover:scale-105 transition-transform duration-300"
                    >
                      <span className="text-2xl">{getDisasterIcon(tag)}</span>
                      <span className="capitalize">{tag.replace(/_/g, ' ')}</span>
                    </Badge>
                  ))}
                </div>

                <div className="bg-muted/50 p-3 rounded-md">
                  <p className="text-sm font-medium mb-1">Summary:</p>
                  <p className="text-sm">{analysisResult.summary}</p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium">Confidence</p>
                    <span className="text-xs font-medium">{analysisResult.confidence}%</span>
                  </div>
                  <Progress value={analysisResult.confidence} className="h-2" />
                </div>

                <div>
                  <p className="text-sm font-medium mb-1">Risk Assessment:</p>
                  <Badge
                    variant={
                      analysisResult.severity === "High"
                        ? "destructive"
                        : analysisResult.severity === "Medium"
                        ? "outline"
                        : "secondary"
                    }
                  >
                    {analysisResult.severity} Risk
                  </Badge>
                </div>

                <Separator />

                <div>
                  <p className="text-sm font-medium mb-2">Recommended Actions:</p>
                  <ul className="space-y-1">
                    {analysisResult.recommendations.map((rec: string, i: number) => (
                      <li key={i} className="text-sm flex items-start gap-2">
                        <span className="text-primary mt-1">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center text-muted-foreground">
                <Clock className="h-10 w-10 mb-2" />
                <p>No analysis results yet</p>
                <p className="text-sm">Upload an image and analyze it to see results here</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}