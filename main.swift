import AVFoundation
import Vision
import CoreMedia
import Foundation

class VideoProcessor {
    private var assetReader: AVAssetReader?
    private let handPoseRequest = VNDetectHumanHandPoseRequest()
    private let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
    private var csvLines: [String] = []
    private let jointNames = ["wrist", "thumbCMC", "thumbMCP", "thumbIP", "thumbTip", "indexFingerMCP", "indexFingerPIP", "indexFingerDIP", "indexFingerTip", "middleFingerMCP", "middleFingerPIP", "middleFingerDIP", "middleFingerTip", "ringFingerMCP", "ringFingerPIP", "ringFingerDIP", "ringFingerTip", "pinkyMCP", "pinkyPIP", "pinkyDIP", "pinkyTip"]
    private let bodyIds = ["nose", "neck", "rightEye", "leftEye", "rightEar", "leftEar", "rightShoulder", "leftShoulder", "rightElbow", "leftElbow", "rightWrist", "leftWrist"]
    private let hands = ["left", "right"]
    let jointKeyMapping: [String: String] = [
        "VNHLKWRI": "wrist",
        "VNHLKTCMC": "thumbCMC",
        "VNHLKTMP": "thumbMCP",
        "VNHLKTIP": "thumbIP",
        "VNHLKTTIP": "thumbTip",
        "VNHLKIMCP": "indexFingerMCP",
        "VNHLKIPIP": "indexFingerPIP",
        "VNHLKIDIP": "indexFingerDIP",
        "VNHLKITIP": "indexFingerTip",
        "VNHLKMMCP": "middleFingerMCP",
        "VNHLKMPIP": "middleFingerPIP",
        "VNHLKMDIP": "middleFingerDIP",
        "VNHLKMTIP": "middleFingerTip",
        "VNHLKRMCP": "ringFingerMCP",
        "VNHLKRPIP": "ringFingerPIP",
        "VNHLKRDIP": "ringFingerDIP",
        "VNHLKRTIP": "ringFingerTip",
        "VNHLKPMCP": "pinkyMCP",
        "VNHLKPPIP": "pinkyPIP",
        "VNHLKPDIP": "pinkyDIP",
        "VNHLKPTIP": "pinkyTip"
    ]
    private let bodyKeyMap: [String: String] = [
        "right_forearm_joint": "rightElbow",
        "head_joint": "nose",
        "right_eye_joint": "rightEye",
        "right_ear_joint": "rightEar",
        "neck_1_joint": "neck",
        "left_forearm_joint": "leftElblow",
        "left_eye_joint": "leftEye",
        "left_ear_joint": "leftEar",
        "left_shoulder_1_joint": "leftShoulder",
        "left_hand_joint": "leftWrist",
        "right_hand_joint": "rightWrist"
    ]
    private var outputPath: String

    init(videoURL: URL, outputCSVPath: String) {
        self.outputPath = outputCSVPath
        handPoseRequest.maximumHandCount = 2
        setupCSVHeaders()
        setupAssetReader(url: videoURL)
        
    }
    
    private func setupCSVHeaders() {
        var headers = [String]()
        for hand in hands {
            for joint in jointNames {
                headers.append("\(joint)_\(hand)_X")
                headers.append("\(joint)_\(hand)_Y")
            }
            for bodyId in bodyIds {
                headers.append("\(bodyId)_X")
                headers.append("\(bodyId)_Y")
            }
        }
        csvLines.append(headers.joined(separator: ","))
    }

    private func setupAssetReader(url: URL) {
        let asset = AVAsset(url: url)
        do {
            assetReader = try AVAssetReader(asset: asset)
            guard let videoTrack = asset.tracks(withMediaType: .video).first else {
                print("No video tracks found in the asset.")
                return
            }
            let trackOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [String(kCVPixelBufferPixelFormatTypeKey): Int(kCVPixelFormatType_32BGRA)])
            assetReader?.add(trackOutput)
            assetReader?.startReading()
            processFrames(trackOutput: trackOutput)
        } catch {
            print("Failed to set up the asset reader: \(error)")
        }
    }

    private func processFrames(trackOutput: AVAssetReaderTrackOutput) {
        while let sampleBuffer = trackOutput.copyNextSampleBuffer(), assetReader?.status == .reading {
            let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
            do {
                try handler.perform([self.handPoseRequest, self.bodyPoseRequest])
                guard let bodyResults = self.bodyPoseRequest.results, let handResults = self.handPoseRequest.results else {
                    print("No results")
                    continue
                }
                processObservations(body_observations: bodyResults, hand_observations: handResults)
            } catch {
                print("Failed to perform Vision request: \(error)")
            }
        }
        if assetReader?.status == .completed {
            print("Processing completed.")
            writeCSV()
        }
    }
  
    private func processObservations(body_observations: [VNHumanBodyPoseObservation], hand_observations: [VNHumanHandPoseObservation]) {
        var dataRow = [String: Float]()
        // Initialize all body data points to zero
        for hand in hands {
            for joint in jointNames {
                dataRow["\(joint)_\(hand)_X"] = 0
                dataRow["\(joint)_\(hand)_Y"] = 0
            }
        }
        for bodyId in bodyIds {
            dataRow["\(bodyId)_X"] = 0
            dataRow["\(bodyId)_Y"] = 0
        }
        // Process each body observation (typically only one body detected)
        for observation in body_observations {
            guard let recognizedPoints = try? observation.recognizedPoints(.all) else {
                print("Could not recognize any body points in this frame.")
                continue
            }

            for (key, point) in recognizedPoints where point.confidence > 0.5 {
                let bodyPart = bodyKeyMap[key.rawValue.rawValue] ?? key.rawValue.rawValue
                let xKey = "\(bodyPart)_X"
                let yKey = "\(bodyPart)_Y"
                dataRow[xKey] = Float(point.location.x)
                dataRow[yKey] = Float(point.location.y)
            }
        }
        for observation in hand_observations {
            let chirality = observation.chirality.rawValue == -1 ? "left" : "right"
            guard let allFingerPoints = try? observation.recognizedPoints(.all) else {
                print("Could not recognize any points in this frame.")
                return
            }
            
            for (joint, point) in allFingerPoints where point.confidence > 0.5 {
                let jointName = jointKeyMapping[joint.rawValue.rawValue] ?? joint.rawValue.rawValue
                let keyX = "\(jointName)_\(chirality)_X"
                let keyY = "\(jointName)_\(chirality)_Y"
                dataRow[keyX] = Float(point.location.x)
                dataRow[keyY] = Float(point.location.y)
            }
        }
        // Convert row dictionary to CSV format
        let sortedKeys = csvLines.first!.split(separator: ",").map(String.init)
        let sortedValues = sortedKeys.map { key -> String in
            String(format: "%.3f", dataRow[key]!)
        }
        let line = sortedValues.joined(separator: ",")
        csvLines.append(line)
    }

      private func writeCSV() {
          let filePath = (self.outputPath as NSString).expandingTildeInPath
          let fileURL = URL(fileURLWithPath: filePath)
          
          do {
              try csvLines.joined(separator: "\n").write(to: fileURL, atomically: true, encoding: .utf8)
              print("CSV file written: \(filePath)")
          } catch {
              print("Error writing CSV file: \(error)")
          }
      }
}

func processAllVideos(in directoryPath: String) {
    let fileManager = FileManager.default
    let directoryURL = URL(fileURLWithPath: directoryPath)
    
    do {
        let contents = try fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
        let videoFiles = contents.filter { $0.pathExtension == "mp4" } // Adjust this filter based on your video file types
        
        for videoFile in videoFiles {
            let csvFileName = videoFile.deletingPathExtension().lastPathComponent + ".csv"
            let outputCSVPath = directoryURL.appendingPathComponent(csvFileName).path
            
            print("Processing video: \(videoFile.lastPathComponent)")
            print("CSV will be saved to: \(outputCSVPath)")
            
            let processor = VideoProcessor(videoURL: videoFile, outputCSVPath: outputCSVPath)
        }
    } catch {
        print("Error while enumerating files \(directoryURL.path): \(error.localizedDescription)")
    }
}

// Example usage
processAllVideos(in: "/Users/science/school/Computer Vision/dalmatian/data/train")
