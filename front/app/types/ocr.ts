export interface Character {
  granulate_symbol: string
  latin_equivalent: string
  confidence: number
}

export interface OCRResponse {
  image_id: string
  text: string
  average_confidence: number
  processing_time: number
  timestamp: string
  characters: Character[]
}