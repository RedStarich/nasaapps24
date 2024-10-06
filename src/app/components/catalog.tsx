'use client'
import { useEffect, useState } from "react"
import { json } from "stream/consumers"

export default function Catalog(){

    const [kek, setKek] = useState(' loading')
    
    useEffect(()=>{
        fetch('http://127.0.0.1:8080/api/data')
        .then((response)=>response.json()
        .then((data)=>console.log(data.message))
    )
    },[])
    
    return(
        <div>
            {kek}
        </div>
    )
}