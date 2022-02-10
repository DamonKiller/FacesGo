package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"

	"database/sql"

	"github.com/Kagami/go-face"
	_ "github.com/mattn/go-sqlite3"
	"gocv.io/x/gocv"
)

const (
	dbPath          = "testdata/faces.db"
	defaultPicsPath = "testdata/images/"
)

var db *sql.DB
var rec *face.Recognizer

//opencv,dlib
var defaultMode, facesPath string
var peoplesName []string

func init() {
	defaultMode = "dlib"

	//TODO 把人脸库的数据加载出来,供下面识别用
	db, _ = sql.Open("sqlite3", dbPath)

	if defaultMode == "dlib" {
		res, _ := db.Prepare("select Name,Descriptor from t_Portrait where Descriptor<>'' and name <>'' ")
		rows, _ := res.Query()
		index := 0
		var peoplesId []int32
		var peoplesDescriptor []face.Descriptor
		for rows.Next() {
			var Name, Descriptor string
			rows.Scan(&Name, &Descriptor)

			var faceinfo face.Descriptor
			dec := gob.NewDecoder(bytes.NewBufferString(Descriptor))
			dec.Decode(&faceinfo)

			peoplesDescriptor = append(peoplesDescriptor, faceinfo)
			peoplesName = append(peoplesName, Name)
			peoplesId = append(peoplesId, int32(index))
			index += 1
		}

		//配置dlib
		rec, _ = face.NewRecognizer("testdata/models/")
		rec.SetSamples(peoplesDescriptor, peoplesId)
	}
}

//go run main.go /{指定目录}
func main() {
	dir := defaultPicsPath
	if len(os.Args) > 2 {
		dir = os.Args[1]
		defaultMode = os.Args[2]
	} else if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	facesPath = dir + "/faces"
	fmt.Println("开始检测目录:" + dir)
	rangeDir(dir)
}

//遍历目录下的所有文件
func rangeDir(dir string) {
	rd, err := ioutil.ReadDir(dir)
	if err != nil {
		fmt.Println("目录读取失败", dir)
		return
	}

	for _, fi := range rd {
		if fi.IsDir() {
			fmt.Println("dir:" + fi.Name())
			rangeDir(dir + "/" + fi.Name())
		} else {
			fileExt := path.Ext(dir + "/" + fi.Name())
			if strings.ToLower(fileExt) != ".jpeg" && strings.ToLower(fileExt) != ".jpg" {
				fmt.Println("not support ext:", fi.Name())
				continue
			}

			fmt.Println("start --> ", dir+"/"+fi.Name())
			if defaultMode == "dlib" {
				findFaceByDLIB(dir, fi.Name(), fileExt)
			} else {
				findFaceByOPENCV(dir, fi.Name(), fileExt)
			}
		}
	}
}

func findFaceByOPENCV(dir string, fn string, fileExt string) {
	img := gocv.IMRead(dir+"/"+fn, gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Error reading image from:", fn)
		return
	}

	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()
	if !classifier.Load("testdata/haar/haarcascade_frontalface_alt.xml") {
		fmt.Println("Error reading cascade file")
		return
	}

	faces := classifier.DetectMultiScale(img)
	fmt.Println("faces count: ", len(faces))
	for i, f := range faces {
		faceinfo, _ := json.Marshal(f)
		fmt.Println(string(faceinfo))
		insertFace(i, f.Min.X, f.Min.Y, f.Max.X, f.Max.Y, dir, fn, fileExt, "")
	}
}

func findFaceByDLIB(dir string, fn string, fileExt string) {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println("findFaceByDLIB error", err)
		}
	}()

	faces, _ := rec.RecognizeFile(dir + "/" + fn)
	fmt.Println("face count:", len(faces))
	for i, f := range faces {
		var buff bytes.Buffer
		enc := gob.NewEncoder(&buff)
		enc.Encode(f.Descriptor)
		insertFace(i, f.Rectangle.Min.X, f.Rectangle.Min.Y, f.Rectangle.Max.X, f.Rectangle.Max.Y, dir, fn, fileExt, string(buff.Bytes()))

		//识别出是具体某人
		if len(peoplesName) > 0 {
			arrid := rec.Classify(f.Descriptor)
			if arrid != -1 && len(peoplesName) > arrid {
				fmt.Println("这人应该是:", peoplesName[arrid])
			}
		}
	}
}

func insertFace(i, x0, y0, x1, y1 int, dir, fn, fileExt string, faceInfo string) {
	src := dir + "/" + fn
	fIn, _ := os.Open(src)
	defer fIn.Close()
	dst := facesPath + strings.Replace(fn, fileExt, "_"+strconv.Itoa(i), 1) + fileExt
	fOut, _ := os.Create(dst)
	defer fOut.Close()
	err := clip(fIn, fOut, x0, y0, x1, y1, 100)
	if err != nil {
		fmt.Println("脸部截取失败." + fn)
	}

	//插入数据
	newfile, _ := os.Stat(dst)
	stmt, _ := db.Prepare("INSERT INTO t_Portrait(fullName,fileSize,sourceFile,isCover,isHidden,isDel,Descriptor) values(?,?,?,?,?,?,?)")
	res, err := stmt.Exec(dst, newfile.Size(), fn, 0, 0, 0, faceInfo)
	if err != nil {
		fmt.Println("DB insert Error.", err, res)
	}
}

//从原图中截出人头,并且保存到新目录下
func clip(in io.Reader, out io.Writer, x0, y0, x1, y1, quality int) error {
	origin, fm, err := image.Decode(in)
	if err != nil {
		return err
	}
	switch fm {
	case "jpeg":
		img := origin.(*image.YCbCr)
		subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.YCbCr)
		return jpeg.Encode(out, subImg, &jpeg.Options{quality})
	case "jpg":
		img := origin.(*image.YCbCr)
		subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.YCbCr)
		return jpeg.Encode(out, subImg, &jpeg.Options{quality})
	default:
		return errors.New("ERROR FORMAT")
	}
}
