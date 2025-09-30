"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Shield, Search, Filter, Calendar, ArrowLeft, Eye, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import Link from "next/link"

// Mock data for incident history
const mockIncidents = [
  {
    id: "INC-001",
    thumbnail: "/placeholder.svg?height=80&width=120",
    userId: "user123",
    date: "2025-05-10T14:30:00Z",
    type: "Road Accident",
    severity: "Moderate",
    summary:
      "Road traffic accident involving two vehicles at an intersection. Moderate damage with potential fluid leakage.",
    details: {
      summary:
        "This image appears to show a road traffic accident involving two vehicles. The collision seems to have occurred at an intersection during nighttime. There is moderate damage to both vehicles, with the front of one car significantly impacted. Several people are gathered around the scene, possibly including emergency responders.",
      details:
        "The accident appears to have resulted in moderate severity damage. There are visible signs of airbag deployment in at least one vehicle, suggesting a significant impact. The positioning of the vehicles indicates a possible T-bone collision at the intersection. There is potential hazard from leaked fluids on the road surface, which could pose slip risks or fire hazards if ignition sources are present.",
      severity: "Moderate",
      confidence: 92,
      timestamp: "2025-05-10T14:30:00Z",
      tags: ["Road Accident", "Nighttime", "Multiple Vehicles", "Crowd Present", "Police Present"],
      recommendations: [
        "Ensure emergency services have been notified",
        "Establish a safe perimeter around the vehicles",
        "Check for fuel or fluid leakage",
        "Direct traffic away from the scene",
        "Gather witness statements if possible",
      ],
    },
  },
  {
    id: "INC-002",
    thumbnail: "/placeholder.svg?height=80&width=120",
    userId: "user456",
    date: "2025-05-09T10:15:00Z",
    type: "Building Fire",
    severity: "High",
    summary: "Commercial building fire with visible flames from multiple windows. High risk of structural collapse.",
    details: {
      summary:
        "This image shows a commercial building fire with visible flames emerging from multiple windows on the second and third floors. Thick black smoke is billowing from the structure, indicating a well-established fire with potentially toxic combustion products. Emergency responders appear to be on scene with equipment deployed.",
      details:
        "The fire appears to be in an advanced stage with significant involvement of at least two floors. The dark color and volume of smoke suggest synthetic materials are burning, which typically produce toxic gases. There are signs of potential structural compromise based on the fire's intensity and duration. Adjacent buildings may be at risk if the fire continues to grow or if wind conditions change.",
      severity: "High",
      confidence: 95,
      timestamp: "2025-05-09T10:15:00Z",
      tags: ["Building Fire", "Commercial Structure", "Multiple Floors", "Heavy Smoke", "Fire Department Present"],
      recommendations: [
        "Maintain safe distance due to potential structural collapse",
        "Evacuate adjacent buildings in fire path",
        "Monitor for falling debris",
        "Consider air quality warnings for downwind areas",
        "Establish multiple water supply points for extended operations",
      ],
    },
  },
  {
    id: "INC-003",
    thumbnail: "/placeholder.svg?height=80&width=120",
    userId: "Anonymous",
    date: "2025-05-08T18:45:00Z",
    type: "Flooding",
    severity: "Medium",
    summary: "Urban street flooding with water approximately 2 feet deep. Multiple stranded vehicles visible.",
    details: {
      summary:
        "This image shows urban street flooding with water approximately 2 feet deep covering a residential or commercial street. Multiple vehicles appear to be stranded in the water, and some pedestrians can be seen attempting to navigate through the flooded area. The flooding appears to extend for at least several blocks.",
      details:
        "The water level is significant enough to disable most passenger vehicles and potentially enter ground-level structures. The water appears to be flowing slowly rather than rapidly, suggesting drainage system overload rather than flash flooding. There are indications of contamination in the water, likely from road pollutants and possibly sewer backup. Electrical hazards may exist from submerged power lines or junction boxes.",
      severity: "Medium",
      confidence: 89,
      timestamp: "2025-05-08T18:45:00Z",
      tags: ["Flooding", "Urban", "Stranded Vehicles", "Infrastructure Damage", "Residential Area"],
      recommendations: [
        "Avoid driving or walking through floodwaters",
        "Be alert for downed power lines and electrical hazards",
        "Monitor for rising water levels",
        "Consider water contamination risks",
        "Check basement flooding in nearby structures",
      ],
    },
  },
  {
    id: "INC-004",
    thumbnail: "/placeholder.svg?height=80&width=120",
    userId: "user789",
    date: "2025-05-07T09:20:00Z",
    type: "Hazardous Materials",
    severity: "Critical",
    summary: "Chemical spill from overturned tanker truck. Visible vapor cloud forming. Immediate evacuation required.",
    details: {
      summary:
        "This image shows a hazardous materials incident involving an overturned tanker truck with visible leakage of chemicals. A vapor cloud is forming around the spill area, and the incident appears to be on a major roadway. Emergency responders in protective gear are visible establishing a perimeter.",
      details:
        "The chemical release appears significant based on the size of the vapor cloud and the rate of spread. The tanker markings indicate a corrosive and toxic material that poses both immediate and prolonged exposure risks. Wind direction is carrying the vapor cloud toward what appears to be a populated area. The integrity of the tanker container is compromised in at least two locations, suggesting multiple release points.",
      severity: "Critical",
      confidence: 97,
      timestamp: "2025-05-07T09:20:00Z",
      tags: ["Hazardous Materials", "Chemical Spill", "Transportation Incident", "Vapor Release", "Evacuation Needed"],
      recommendations: [
        "Immediate evacuation of at least 1/2 mile downwind",
        "Establish decontamination zones",
        "Approach only with appropriate protective equipment",
        "Notify downstream water authorities if near waterways",
        "Implement public alert system for affected areas",
        "Monitor air quality with specialized equipment",
      ],
    },
  },
  {
    id: "INC-005",
    thumbnail: "/placeholder.svg?height=80&width=120",
    userId: "user321",
    date: "2025-05-06T14:10:00Z",
    type: "Structural Collapse",
    severity: "High",
    summary: "Partial collapse of multi-story building. Multiple victims potentially trapped in debris.",
    details: {
      summary:
        "This image shows a partial collapse of a multi-story building, affecting approximately 30% of the structure. The collapse appears to have occurred on one side of the building, creating a significant debris field. There are indications of people potentially trapped within the collapsed portion.",
      details:
        "The structural failure appears to have affected multiple floors simultaneously, suggesting a potential foundation or load-bearing wall failure. The remaining structure shows signs of instability with visible cracks and deformation. The debris field contains large concrete sections, structural steel, and building contents, creating complex void spaces where survivors might be located. Dust is still visible in the air, indicating a recent collapse event.",
      severity: "High",
      confidence: 94,
      timestamp: "2025-05-06T14:10:00Z",
      tags: ["Structural Collapse", "Urban", "Entrapment", "Multi-story Building", "Search and Rescue Needed"],
      recommendations: [
        "Deploy urban search and rescue teams immediately",
        "Establish collapse zones around remaining unstable structure",
        "Implement specialized listening equipment for victim detection",
        "Secure utilities to prevent secondary hazards",
        "Prepare for extended rescue operations",
        "Consider heavy equipment needs for debris removal",
      ],
    },
  },
]

export default function AdminDashboard() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedSeverity, setSelectedSeverity] = useState("")
  const [selectedIncident, setSelectedIncident] = useState<any | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 4

  // Filter incidents based on search term and severity
  const filteredIncidents = mockIncidents.filter((incident) => {
    const matchesSearch =
      incident.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      incident.userId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      incident.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      incident.summary.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesSeverity = selectedSeverity === "" || incident.severity === selectedSeverity

    return matchesSearch && matchesSeverity
  })

  // Paginate incidents
  const totalPages = Math.ceil(filteredIncidents.length / itemsPerPage)
  const paginatedIncidents = filteredIncidents.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString() + " " + date.toLocaleTimeString()
  }

  return (
    <div className="min-h-screen bg-dark-bg p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-accent-orange mr-2" />
              <h1 className="text-3xl font-bold">Admin Panel – Incident History</h1>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Home
              </Link>
            </Button>
          </div>
          <p className="text-muted-foreground">View and manage all incident analyses</p>
        </motion.div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Search and Filter</CardTitle>
            <CardDescription>Find specific incidents by ID, user, type, or description</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search incidents..."
                  className="pl-8"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <div className="flex gap-4">
                <div className="w-40">
                  <Select value={selectedSeverity} onValueChange={setSelectedSeverity}>
                    <SelectTrigger>
                      <SelectValue placeholder="Severity" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Severities</SelectItem>
                      <SelectItem value="Low">Low</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Moderate">Moderate</SelectItem>
                      <SelectItem value="High">High</SelectItem>
                      <SelectItem value="Critical">Critical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button variant="outline" size="icon">
                  <Filter className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon">
                  <Calendar className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Incident History</CardTitle>
            <CardDescription>{filteredIncidents.length} incidents found</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Image</TableHead>
                    <TableHead>ID</TableHead>
                    <TableHead>User</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Severity</TableHead>
                    <TableHead>Summary</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedIncidents.length > 0 ? (
                    paginatedIncidents.map((incident) => (
                      <TableRow key={incident.id}>
                        <TableCell>
                          <img
                            src={incident.thumbnail || "/placeholder.svg"}
                            alt={`Incident ${incident.id}`}
                            className="w-20 h-14 object-cover rounded-md"
                          />
                        </TableCell>
                        <TableCell className="font-medium">{incident.id}</TableCell>
                        <TableCell>{incident.userId}</TableCell>
                        <TableCell>{formatDate(incident.date)}</TableCell>
                        <TableCell>{incident.type}</TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              incident.severity === "Critical"
                                ? "destructive"
                                : incident.severity === "High"
                                  ? "default"
                                  : incident.severity === "Medium" || incident.severity === "Moderate"
                                    ? "secondary"
                                    : "outline"
                            }
                          >
                            {incident.severity}
                          </Badge>
                        </TableCell>
                        <TableCell className="max-w-xs truncate">{incident.summary}</TableCell>
                        <TableCell className="text-right">
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="ghost" size="sm" onClick={() => setSelectedIncident(incident)}>
                                <Eye className="h-4 w-4 mr-1" />
                                View
                              </Button>
                            </DialogTrigger>
                            <DialogContent className="max-w-3xl">
                              <DialogHeader>
                                <DialogTitle>Incident Details: {selectedIncident?.id}</DialogTitle>
                                <DialogDescription>Full analysis report for this incident</DialogDescription>
                              </DialogHeader>

                              {selectedIncident && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                                  <div>
                                    <img
                                      src={selectedIncident.thumbnail || "/placeholder.svg"}
                                      alt={`Incident ${selectedIncident.id}`}
                                      className="w-full h-48 object-cover rounded-lg mb-4"
                                    />

                                    <div className="space-y-3">
                                      <div>
                                        <h4 className="text-sm font-medium text-muted-foreground">ID</h4>
                                        <p>{selectedIncident.id}</p>
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-medium text-muted-foreground">User</h4>
                                        <p>{selectedIncident.userId}</p>
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-medium text-muted-foreground">Date</h4>
                                        <p>{formatDate(selectedIncident.date)}</p>
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-medium text-muted-foreground">Type</h4>
                                        <p>{selectedIncident.type}</p>
                                      </div>
                                      <div>
                                        <h4 className="text-sm font-medium text-muted-foreground">Severity</h4>
                                        <Badge
                                          variant={
                                            selectedIncident.severity === "Critical"
                                              ? "destructive"
                                              : selectedIncident.severity === "High"
                                                ? "default"
                                                : selectedIncident.severity === "Medium" ||
                                                    selectedIncident.severity === "Moderate"
                                                  ? "secondary"
                                                  : "outline"
                                          }
                                        >
                                          {selectedIncident.severity}
                                        </Badge>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="space-y-4">
                                    <div>
                                      <h3 className="text-lg font-medium mb-2">Analysis Summary</h3>
                                      <p className="text-sm">{selectedIncident.details.summary}</p>
                                    </div>

                                    <Separator />

                                    <div>
                                      <h3 className="text-lg font-medium mb-2">Detailed Analysis</h3>
                                      <p className="text-sm">{selectedIncident.details.details}</p>
                                    </div>

                                    <Separator />

                                    <div>
                                      <h4 className="font-medium mb-2">Technical Data</h4>
                                      <div className="grid grid-cols-2 gap-3">
                                        <div>
                                          <p className="text-sm text-muted-foreground">Severity</p>
                                          <p className="font-medium">{selectedIncident.details.severity}</p>
                                        </div>
                                        <div>
                                          <p className="text-sm text-muted-foreground">Confidence</p>
                                          <p className="font-medium">{selectedIncident.details.confidence}%</p>
                                        </div>
                                        <div>
                                          <p className="text-sm text-muted-foreground">Analysis Time</p>
                                          <p className="font-medium">
                                            {formatDate(selectedIncident.details.timestamp)}
                                          </p>
                                        </div>
                                      </div>
                                    </div>

                                    <Separator />

                                    <div>
                                      <h4 className="font-medium mb-2">Tags</h4>
                                      <div className="flex flex-wrap gap-2">
                                        {selectedIncident.details.tags.map((tag: string, index: number) => (
                                          <Badge key={index} variant="secondary">
                                            {tag}
                                          </Badge>
                                        ))}
                                      </div>
                                    </div>

                                    <Separator />

                                    <div>
                                      <h4 className="font-medium mb-2">Recommendations</h4>
                                      <ul className="space-y-2">
                                        {selectedIncident.details.recommendations.map((rec: string, index: number) => (
                                          <li key={index} className="text-sm">
                                            • {rec}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </DialogContent>
                          </Dialog>
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={8} className="h-24 text-center">
                        No incidents found.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-end space-x-2 py-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                  <span className="sr-only">Previous Page</span>
                </Button>
                <div className="text-sm text-muted-foreground">
                  Page {currentPage} of {totalPages}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight className="h-4 w-4" />
                  <span className="sr-only">Next Page</span>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
