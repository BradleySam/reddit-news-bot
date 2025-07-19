package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/mmcdole/gofeed"
)

// Story represents a Reddit news story
type Story struct {
	Title string
	Link  string
}

// SlackPayload defines the message format for Slack webhook
type SlackPayload struct {
	Text string `json:"text"`
}

// Constants
const (
	redditRSS    = "https://www.reddit.com/r/news/top/.rss?t=day"
	hfModelURL   = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
	summaryLimit = 5
)

func main() {
	// Load environment variables from .env
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found — assuming environment variables are already set.")
	}

	// Get API credentials
	slackWebhook := os.Getenv("SLACK_WEBHOOK_URL")
	hfAPIKey := os.Getenv("HUGGINGFACE_API_KEY")

	if slackWebhook == "" || hfAPIKey == "" {
		log.Fatal("Missing SLACK_WEBHOOK_URL or HUGGINGFACE_API_KEY in environment")
	}

	// Send the date as the first Slack message
	currentDate := time.Now().Format("🗓️ January 2, 2006")
	err = postToSlack(slackWebhook, currentDate)
	if err != nil {
		log.Fatalf("Error posting date to Slack: %v", err)
	}

	// Fetch top Reddit news stories
	stories, err := fetchTopStories(summaryLimit)
	if err != nil {
		log.Fatalf("Failed to fetch stories: %v", err)
	}

	var wg sync.WaitGroup

	// Launch goroutines for each story
	for _, story := range stories {
		wg.Add(1)
		go func(s Story) {
			defer wg.Done()
			processStory(s, hfAPIKey, slackWebhook)
		}(story)
	}

	// Wait for all summaries to be processed
	wg.Wait()
}

// processStory handles summarization and Slack posting for a single story
func processStory(story Story, hfAPIKey, slackWebhook string) {
	// Combine title and link for summarization input
	text := fmt.Sprintf("%s - %s", story.Title, story.Link)

	// Summarize the story using Hugging Face
	summary, err := summarizeWithHuggingFace(hfAPIKey, text)
	if err != nil {
		log.Printf("Error summarizing '%s': %v", story.Title, err)
		return
	}

	// Format Slack message (no separator line, no links)
	message := fmt.Sprintf("*Title:* %s\n> %s", story.Title, summary)

	// Send to Slack
	err = postToSlack(slackWebhook, message)
	if err != nil {
		log.Printf("Error posting to Slack: %v", err)
	}
}

// fetchTopStories pulls N top stories from Reddit's RSS feed
func fetchTopStories(limit int) ([]Story, error) {
	fp := gofeed.NewParser()
	feed, err := fp.ParseURL(redditRSS)
	if err != nil {
		return nil, err
	}

	var stories []Story
	for i, item := range feed.Items {
		if i >= limit {
			break
		}
		stories = append(stories, Story{
			Title: item.Title,
			Link:  item.Link,
		})
	}
	return stories, nil
}

// summarizeWithHuggingFace uses the Hugging Face inference API to summarize text
func summarizeWithHuggingFace(apiKey, text string) (string, error) {
	body, _ := json.Marshal(map[string]string{"inputs": text})

	req, err := http.NewRequest("POST", hfModelURL, bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 40 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result []map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result) > 0 && result[0]["summary_text"] != "" {
		return result[0]["summary_text"], nil
	}

	return "Summary unavailable", nil
}

// postToSlack sends a formatted message to the Slack webhook
func postToSlack(webhookURL, message string) error {
	payload := SlackPayload{Text: message}
	data, _ := json.Marshal(payload)

	resp, err := http.Post(webhookURL, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Slack responded with status: %v", resp.Status)
	}
	return nil
}
