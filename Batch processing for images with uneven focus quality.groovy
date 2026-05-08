//Batch processing for images with uneven focus quality
//Yanyan Chen (05/06/2026; Columbia University; yc4569@cumc.columbia.edu)

//Import OpenCV and Qupath OpenCV library functions for image processing

import org.bytedeco.opencv.global.opencv_imgproc
import org.bytedeco.opencv.global.opencv_core
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_core.Size
import qupath.opencv.tools.OpenCVTools

//Define Functions (2 functions here)
//Function: FocusScore calculation: using the variance of Laplacian

def calculateFocusScoreForTiles() {

    def imageData = getCurrentImageData()
    def server = imageData.getServer()
    def tiles = getAnnotationObjects()

    for (tile in tiles) {

        def roi = tile.getROI()
        def request = RegionRequest.createInstance(server.getPath(), 1.0, roi)
        def img = server.readRegion(request)

        def mat = OpenCVTools.imageToMat(img)

        //Channels: blue and green channels
        Mat green = new Mat()
        Mat blue = new Mat()

        opencv_core.extractChannel(mat, blue, 0)
        opencv_core.extractChannel(mat, green, 1)

        //Combine blue and green channels
        Mat combined = new Mat()
        opencv_core.addWeighted(green, 0.5, blue, 0.5, 0, combined)

        //Blur
        opencv_imgproc.GaussianBlur(combined, combined, new Size(3,3), 0)

        //Laplacian
        Mat lap = new Mat()
        opencv_imgproc.Laplacian(combined, lap, opencv_core.CV_64F)

        //Stats
        Mat mean = new Mat()
        Mat stddev = new Mat()
        opencv_core.meanStdDev(lap, mean, stddev)

        double std = stddev.createBuffer().get(0)
        double variance = std * std

        tile.getMeasurementList().put("FocusScore", variance)
    }

    fireHierarchyUpdate()

    println "FocusScore calculated for " + tiles.size() + " tiles"
}


//Function: Select in-focus tiles that has focus score higher than a threshold value

def markInFocusTiles(double thresholdValue) {
    def tiles = getAnnotationObjects()

    // Create the InFocus class
    def goodClass = getPathClass("InFocus") ?: addPathClass("InFocus")

    def inFocusTiles = []

    tiles.each { tile ->
        def score = tile.getMeasurementList().get("FocusScore")
        if (score != null && score > thresholdValue) {
            tile.setPathClass(goodClass)
            inFocusTiles << tile
        }
    }

    fireHierarchyUpdate()

    return inFocusTiles
}

//-----------------------------------------------------------------
//Main batch pipeline

def project = getProject()

for (entry in project.getImageList()) {

    //Print the file name that is going to be processed
    println "Processing: " + entry.getImageName()   

    def imageData = entry.readImageData()
    setBatchProjectAndImage(project, imageData)

    //Reset and select the image type
    clearAllObjects()
    setImageType('FLUORESCENCE')
    resetSelection()

    //Well area detection using the classifier "WellArea" saved from pixel classification --> create thresholder 
    createAnnotationsFromPixelClassifier("WellArea", 500000.0, 0.0)

    println "Number of annotations from classifier: " + getAnnotationObjects().size()

    //Cut the WellArea into tiles and remove the parent annotation
    selectObjects(getAnnotationObjects())
    runPlugin('qupath.lib.algorithms.TilerPlugin',
        '{"tileSizeMicrons":1450.0,"trimToROI":true,"makeAnnotations":true,"removeParentAnnotation":true}')

    println "Total tiles AFTER tiling: " + getAnnotationObjects().size()

    //Calculate the FocusScore for each tile
    calculateFocusScoreForTiles()

    // Filter in-focus tiles based on the threshold. Note: threshold = 45000 is an empirical number 
    double threshold = 45000.0
    def inFocusTiles = markInFocusTiles(threshold)

    println "In-focus tiles: " + inFocusTiles.size()

    if (inFocusTiles.isEmpty()) {
        println "No in-focus tiles → skipping image"
        continue
    }

    // Cell detection only on the InFocus tiles
    selectObjects(inFocusTiles)

    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection',
        '{"detectionImage":"Channel 3","requestedPixelSizeMicrons":0.5,"backgroundRadiusMicrons":8.0,"backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":2.0,"minAreaMicrons":25.0,"maxAreaMicrons":200.0,"threshold":2500.0,"watershedPostProcess":true,"cellExpansionMicrons":4.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true}')

    resetSelection()

    //Found out the green neuron cells, cells with red signal and green neuron with red signal from the saved classifiers
    runObjectClassifier("Green_Neuron")
    runObjectClassifier("Red_signal")
    runObjectClassifier("Green_Neuron_with_Red")

    // Save results to the results folder under the Project, each image will have a ImageName.cvs file 
    def resultsDir = buildFilePath(PROJECT_BASE_DIR, "results")
    mkdirs(resultsDir)

    def outPath = buildFilePath(resultsDir, entry.getImageName() + ".csv")

    saveAnnotationMeasurements(outPath)

    entry.saveImageData(imageData)
}

println "Processing finished"